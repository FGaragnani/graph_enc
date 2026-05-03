import json
import gc
import logging
import math
import os
import sys
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import datasets
import evaluate
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.hub import send_example_telemetry

from graph_enc.src.data.en_prom_dataset import PromoterEnhancerDataset
from graph_enc.src.model.model import BertForMaskedLM

check_min_version("4.29.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for initialization. Leave empty to train from scratch.",
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, model type from: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": "Override config when training from scratch, e.g. hidden_size=768,num_hidden_layers=8"},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Config path/name if different from model"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer path/name if different from model"})
    config_file: Optional[str] = field(default=None, metadata={"help": "Config file path."})
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    low_cpu_mem_usage: bool = field(default=False)
    hidden_dropout_prob: Optional[float] = field(default=None)
    attention_probs_dropout_prob: Optional[float] = field(default=None)
    classifier_dropout: float = field(default=0.1)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    dataset_dir: str = field(
        default="scripts/datasets",
        metadata={"help": "Directory containing enhancers.dat and promoters.dat"},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional genome path used to materialize enhancer sequences."},
    )
    target_enhancer_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Optional target enhancer fraction in dataset after undersampling majority class (e.g. 0.5 for balanced)."
        },
    )
    balance_seed: int = field(
        default=42,
        metadata={"help": "Random seed used for deterministic class rebalancing."},
    )
    validation_split_percentage: int = field(default=5)
    max_seq_length: Optional[int] = field(
        default=768,
        metadata={"help": "Tokenizer max length for each chunk (e.g. 768)."},
    )
    chunk_size_bases: int = field(
        default=2000,
        metadata={"help": "Split long promoter/enhancer sequences into chunks of this many bases."},
    )
    perform_kfold: bool = field(
        default=False,
        metadata={"help": "Whether to perform K-Fold cross validation instead of a single train/validation split."},
    )
    max_chunks_per_sample: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on chunks per sequence (keep first N chunks)."},
    )
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    def __post_init__(self):
        if not os.path.isdir(self.dataset_dir):
            raise ValueError(f"--dataset_dir must be an existing directory. Got: {self.dataset_dir}")
        if self.data_path is not None and not os.path.isdir(self.data_path):
            raise ValueError(f"--data_path must be an existing directory. Got: {self.data_path}")
        if self.chunk_size_bases <= 0:
            raise ValueError("--chunk_size_bases must be > 0")
        if self.max_seq_length is not None and self.max_seq_length <= 0:
            raise ValueError("--max_seq_length must be > 0")
        if self.target_enhancer_fraction is not None:
            if not (0.0 < self.target_enhancer_fraction < 1.0):
                raise ValueError("--target_enhancer_fraction must be in (0, 1)")


class PEDiscriminativeDataset(TorchDataset):
    """Dataset exposing sequence + binary label (0 promoter, 1 enhancer)."""

    def __init__(self, base_dataset: PromoterEnhancerDataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.base_dataset.data[idx]
        sequence = self.base_dataset[idx]
        label = 1 if item.is_enhancer() else 0
        return {"sequence": sequence, "labels": label}


class DataCollatorForChunkedPromoterEnhancer:
    """Collator that chunks long sequences and keeps a map chunk->original sample."""

    def __init__(
        self,
        tokenizer,
        max_seq_length: int,
        chunk_size_bases: int = 2000,
        pad_to_max_length: bool = False,
        max_chunks_per_sample: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunk_size_bases = chunk_size_bases
        self.pad_to_max_length = pad_to_max_length
        self.max_chunks_per_sample = max_chunks_per_sample
        self.pad_to_multiple_of = pad_to_multiple_of

    def _split_sequence(self, sequence: str) -> List[str]:
        chunks = [sequence[i : i + self.chunk_size_bases] for i in range(0, len(sequence), self.chunk_size_bases)]
        if not chunks:
            chunks = ["N"]
        if self.max_chunks_per_sample is not None:
            chunks = chunks[: self.max_chunks_per_sample]
        return chunks

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        labels: List[int] = []
        all_chunks: List[str] = []
        chunk_to_sample: List[int] = []

        for sample_idx, feature in enumerate(features):
            label = int(feature["labels"])
            sequence = str(feature["sequence"]).strip().upper()
            labels.append(label)

            if not sequence:
                sequence = "N"

            chunks = self._split_sequence(sequence)
            all_chunks.extend(chunks)
            chunk_to_sample.extend([sample_idx] * len(chunks))

        tokenized = self.tokenizer(
            all_chunks,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length" if self.pad_to_max_length else False,
            return_attention_mask=True,
        )
        batch = self.tokenizer.pad(
            tokenized,
            return_tensors="pt",
            padding="max_length" if self.pad_to_max_length else True,
            max_length=self.max_seq_length if self.pad_to_max_length else None,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch["chunk_to_sample"] = torch.tensor(chunk_to_sample, dtype=torch.long)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class ChunkAveragedCLSClassifier(nn.Module):
    """Encode each chunk, average CLS chunk embeddings per sample, classify."""

    def __init__(self, backbone_mlm: BertForMaskedLM, hidden_size: int, num_labels: int, classifier_dropout: float):
        super().__init__()
        self.backbone = backbone_mlm.bert
        self.num_labels = num_labels
        self.loss_fct = nn.CrossEntropyLoss()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        chunk_to_sample: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if chunk_to_sample is None:
            raise ValueError("chunk_to_sample is required.")

        encoder_outputs, _ = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_all_encoded_layers=False,
        )

        sequence_output = encoder_outputs[-1] if isinstance(encoder_outputs, list) else encoder_outputs
        cls_per_chunk = sequence_output[:, 0, :]

        if labels is not None:
            batch_size = labels.size(0)
        else:
            batch_size = int(chunk_to_sample.max().item()) + 1

        pooled = cls_per_chunk.new_zeros((batch_size, cls_per_chunk.size(-1)))
        pooled.index_add_(0, chunk_to_sample, cls_per_chunk)

        counts = torch.bincount(chunk_to_sample, minlength=batch_size).to(device=cls_per_chunk.device)
        counts = counts.clamp_min(1).unsqueeze(-1).to(dtype=cls_per_chunk.dtype)
        pooled = pooled / counts

        pooled = self.projection(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "--logging_steps" not in sys.argv and "--logging_strategy" not in sys.argv:
        training_args.logging_strategy = "steps"
        training_args.logging_steps = 10
    if "--logging_first_step" not in sys.argv:
        training_args.logging_first_step = True

    # Keep runs stateless by default; we only persist one selected fold below.
    training_args.save_strategy = "no"

    # Trainer must keep sequence column for custom collator.
    training_args.remove_unused_columns = False

    send_example_telemetry("discriminative_promoter_enhancer", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    base_dataset = PromoterEnhancerDataset(
        dir=data_args.dataset_dir,
        genome_data_path=data_args.data_path,
        target_enhancer_fraction=data_args.target_enhancer_fraction,
        balance_seed=data_args.balance_seed,
    )
    dataset = PEDiscriminativeDataset(base_dataset)

    if training_args.do_eval:
        all_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(training_args.seed)).tolist()
        
        split_idx = int(len(all_idx) * (data_args.validation_split_percentage / 100))
        split_idx = min(max(split_idx, 1), max(len(all_idx) - 1, 1))
        if data_args.perform_kfold:
            folds = [
                (all_idx[i * split_idx : (i + 1) * split_idx], all_idx[: i * split_idx] + all_idx[(i + 1) * split_idx :])
                for i in range((len(all_idx) + split_idx - 1) // split_idx)
            ]
            folds = [fold for fold in folds if len(fold[0]) > 0 and len(fold[1]) > 0]
        else:
            eval_idx = all_idx[:split_idx]
            train_idx = all_idx[split_idx:]
            folds = [
                (eval_idx,
                train_idx)
            ]
    else:
        folds = [(list(range(len(dataset))), [])]

    def _resolve_local_path(path_value: Optional[str]) -> Optional[str]:
        if path_value is None:
            return None
        if os.path.isfile(path_value) or os.path.isdir(path_value):
            return os.path.abspath(path_value)
        return path_value

    model_args.config_name = model_args.config_file
    model_args.tokenizer_name = _resolve_local_path(model_args.tokenizer_name)
    model_args.model_name_or_path = _resolve_local_path(model_args.model_name_or_path)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.config_name:
        if os.path.isfile(model_args.config_name):
            with open(model_args.config_name, "r", encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
            config_model_type = model_args.model_type or config_dict.get("model_type")
            if config_model_type is None:
                raise ValueError("Config JSON does not specify model_type and --model_type was not provided.")
            config = CONFIG_MAPPING[config_model_type].from_dict(config_dict)
        else:
            config = AutoConfig.from_pretrained(model_args.config_name, trust_remote_code=True, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)
    else:
        if model_args.model_type is None:
            raise ValueError("When training from scratch, --model_type must be provided.")
        config = CONFIG_MAPPING[model_args.model_type]()
        if model_args.config_overrides is not None:
            config.update_from_string(model_args.config_overrides)

    if model_args.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if model_args.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs)
    else:
        raise ValueError("Tokenizer cannot be created from scratch here. Provide --tokenizer_name.")

    logger.info(f"Vocabulary size: {len(tokenizer)}")

    if model_args.model_name_or_path:
        backbone_mlm = BertForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        backbone_mlm = BertForMaskedLM(config)

    embedding_size = backbone_mlm.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        backbone_mlm.resize_token_embeddings(len(tokenizer))

    if data_args.max_seq_length is None:
        max_seq_length = min(tokenizer.model_max_length, 768)
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    model = ChunkAveragedCLSClassifier(
        backbone_mlm=backbone_mlm,
        hidden_size=config.hidden_size,
        num_labels=2,
        classifier_dropout=model_args.classifier_dropout,
    )
    base_model = copy.deepcopy(model)

    data_collator = DataCollatorForChunkedPromoterEnhancer(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        chunk_size_bases=data_args.chunk_size_bases,
        pad_to_max_length=data_args.pad_to_max_length,
        max_chunks_per_sample=data_args.max_chunks_per_sample,
        pad_to_multiple_of=8 if training_args.fp16 and not data_args.pad_to_max_length else None,
    )

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]
        return {"accuracy": accuracy, "f1": f1}
    
    results = {}

    for fold_idx, fold in enumerate(folds):
        eval_idx, train_idx = fold
        logger.info(
            f"Starting fold {fold_idx + 1}/{len(folds)} with {len(train_idx)} train and {len(eval_idx)} eval samples"
        )

        # Reinitialize model every fold to avoid training-state leakage across folds.
        model = copy.deepcopy(base_model)
        train_dataset = Subset(dataset, train_idx)
        eval_dataset = Subset(dataset, eval_idx) if eval_idx else None
        trainer = None
        train_result = None
        
        if training_args.do_train and data_args.max_train_samples is not None:
            train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), data_args.max_train_samples))))
        if training_args.do_eval and data_args.max_eval_samples is not None and eval_dataset is not None:
            eval_dataset = Subset(eval_dataset, list(range(min(len(eval_dataset), data_args.max_eval_samples))))
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
        )

        try:
            if training_args.do_train:
                checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else last_checkpoint
                train_result = trainer.train(resume_from_checkpoint=checkpoint)

                metrics = train_result.metrics
                metrics["train_samples"] = len(train_dataset)
                train_metric_prefix = "train" if len(folds) == 1 else f"train_fold_{fold_idx}"
                trainer.log_metrics(train_metric_prefix, metrics)
                trainer.save_metrics(train_metric_prefix, metrics)

                if training_args.seed == 42 and fold_idx == 0:
                    trainer.save_model()

            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                metrics = trainer.evaluate()
                metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
                eval_metric_prefix = "eval" if len(folds) == 1 else f"eval_fold_{fold_idx}"
                trainer.log_metrics(eval_metric_prefix, metrics)
                trainer.save_metrics(eval_metric_prefix, metrics)
                results[len(results)] = metrics
        finally:
            if trainer is not None and hasattr(trainer, "model") and trainer.model is not None:
                trainer.model.cpu()
            del trainer
            del model
            del train_dataset
            del eval_dataset
            del train_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    for key, value in results.items():
        logger.info(f"Fold {key}: {value}")

if __name__ == "__main__":
    main()

# DNABERT-2 for Promoter / Enhancer Classification

A reimplementation of [DNABERT-2](https://arxiv.org/abs/2306.15006) trained and evaluated on human (and mouse) genome regulatory elements. We compare two training strategies - domain-specific pretraining on coding sequences followed by fine-tuning vs. fine-tuning from a randomly-initialised encoder - and show they are statistically equivalent.

> AI for Bioinformatics Course

---

## How it works

### Model architecture

We use a 12-layer BERT encoder with a BPE tokeniser (4 096 tokens) built for DNA.

Config lives in [`src/model/bert_config.json`](src/model/bert_config.json): 12 layers, 768 hidden, 12 heads, 3072 FFN, 4'096 vocab.

### Training pipeline

```
Stage 1 (optional)   Masked Language Modelling on human coding sequences
                        src/train/train.py

Stage 2              Binary classification: promoter vs. enhancer
                        src/task/discriminative.py
```

**Classification head (`ChunkAveragedCLSClassifier`)** handles variable-length sequences by:
1. Splitting each sequence into non-overlapping 2'000 bp chunks.
2. Mean-pooling token embeddings within each chunk → one vector per chunk.
3. Mean-pooling chunk vectors → one vector per sequence.
4. A linear projection (768 → 2) produces the final logits.

Promoters are fixed at 2'000 bp (one chunk); enhancers can span up to ~10 kb (≤ 10 chunks).

### Evaluation

- **10-fold cross-validation**, stratified at chromosome level (no leakage between folds).
- **7 random seeds** (42 – 48) per model.
- Metrics: Accuracy, macro-F1, ROC-AUC.
- Statistical equivalence tested with a paired t-test + TOST (± 0.02 margin).
- Zero-shot cross-species evaluation on mouse (GRCm39) without any retraining.

---

## Repository layout

```
graph_enc/
├── src/
│   ├── model/                  # BERT architecture, tokeniser, config
│   │   ├── model.py
│   │   ├── bert_padding.py
│   │   ├── bert_config.json
│   │   └── tokenizer.json
│   ├── data/
│   │   ├── dataset.py          # ChromosomeDataset for pretraining
│   │   └── en_prom_dataset.py  # PromoterEnhancerDataset for fine-tuning
│   ├── train/
│   │   └── train.py            # MLM pretraining loop
│   └── task/
│       ├── discriminative.py   # Fine-tuning & evaluation (main script)
│       └── kmer_lr_baseline.py # k-mer + logistic regression baseline
│
├── scripts/
│   ├── pretrain_lm_cs.sh           # SLURM: pretraining
│   ├── finetune_pe.sh              # SLURM: fine-tune (pretrained init)
│   ├── finetune_scratch_pe.sh      # SLURM: fine-tune (random init)
│   ├── finetune_pe_overlap.sh      # SLURM: fine-tune with overlapping chunks
│   ├── finetune_full_pe.sh         # SLURM: full-dataset fine-tune
│   ├── finetune_full_pe_scratch.sh # SLURM: full-dataset, scratch
│   ├── eval_mouse_pe.sh            # SLURM: zero-shot mouse evaluation
│   └── datasets/
│       ├── promoters.dat           # Human promoters (EPDNew)
│       ├── enhancers.dat           # Human enhancers (ENdb)
│       └── mouse/                  # Mouse equivalents
│
├── data/
│   ├── download_chr_cds.sh         # Download human genome + GTF
│   └── download_mouse_chr_cds.sh   # Download mouse genome + GTF
│
├── logs/
│   ├── pretrain/                   # SLURM stdout/stderr for pretraining runs
│   └── folds/
│       ├── pretrained/             # Per-seed .out files + CI plots
│       ├── scratch/                # Per-seed .out files + CI plots
│       ├── mice/                   # Mouse evaluation logs
│       ├── parse_folds.py          # Parse raw log files → structured metrics
│       ├── compute_ci_from_folds.py    # Confidence intervals (human)
│       ├── compute_ci_from_mice.py     # Confidence intervals (mouse)
│       ├── compare_model_logs_ttest.py # Paired t-test + TOST
│       └── ci_utils.py                 # Shared utilities
│
├── report/
│   ├── report.tex
│   ├── report.pdf
│   └── references.bib
│
└── requirements.txt
```

---

## Setup

```bash
git clone <this-repo>
cd graph_enc
pip install -r requirements.txt
export PYTHONPATH=.
```

Download the genome data (FASTA + GTF):

```bash
bash data/download_chr_cds.sh       1-19 # human
bash data/download_mouse_chr_cds.sh 1-22 # mouse (needed only for cross-species eval)
```

---

## Running the experiments

All scripts are written for SLURM but the underlying `torchrun` / `python` calls can be run locally. The SLURM header lines are just metadata - strip them or submit with `sbatch`.

### Stage 1 - Pretraining (optional)

Skip this if you only want the scratch baseline.

```bash
torchrun --nproc_per_node=4 src/train/train.py \
  --model_type bert \
  --config_name src/model/bert_config.json \
  --tokenizer_name ./src/model \
  --data_path ./data/ \
  --use_cds_mask true \
  --max_seq_length 768 \
  --mlm_probability 0.15 \
  --output_dir ./checkpoints/dnabert2_cs \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-3 \
  --max_steps 40000 \
  --warmup_steps 1200 \
  --weight_decay 1e-5 \
  --adam_beta1 0.9 --adam_beta2 0.98 --adam_eps 1e-6 \
  --bf16 \
  --do_train --do_eval --overwrite_output_dir
```

### Stage 2 - Fine-tuning (7 seeds × 10 folds)

**From scratch** (no pretrained checkpoint):

```bash
for seed in 42 43 44 45 46 47 48; do
  torchrun --nproc_per_node=4 src/task/discriminative.py \
    --model_type bert \
    --config_file ./src/model/bert_config.json \
    --tokenizer_name ./src/model \
    --data_path ./data/ \
    --dataset_dir ./scripts/datasets \
    --target_enhancer_fraction 0.5 \
    --balance_seed ${seed} \
    --max_seq_length 768 \
    --chunk_size_bases 2000 \
    --max_chunks_per_sample 10 \
    --output_dir ./output/scratch/seed_${seed} \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --max_steps 4000 \
    --warmup_steps 400 \
    --weight_decay 1e-5 \
    --bf16 \
    --perform_kfold true \
    --seed ${seed} \
    --do_train --do_eval --overwrite_output_dir
done
```

**With pretrained checkpoint** - same command, add:

```bash
  --model_name_or_path ./checkpoints/dnabert2_cs \
```

### Zero-shot evaluation on mouse

Requires a model trained on the full human dataset (use `finetune_full_pe.sh` / `finetune_full_pe_scratch.sh` first).

```bash
python src/task/discriminative.py \
  --model_name_or_path ./output/full/seed_42 \
  --model_type bert \
  --config_file ./src/model/bert_config.json \
  --tokenizer_name ./src/model \
  --data_path ./data/ \
  --dataset_dir ./scripts/datasets/mouse \
  --is_human false \
  --target_enhancer_fraction 0.5 \
  --balance_seed 42 \
  --max_seq_length 768 \
  --chunk_size_bases 2000 \
  --max_chunks_per_sample 10 \
  --output_dir ./output/mouse_eval \
  --per_device_eval_batch_size 4 \
  --perform_kfold false \
  --do_eval --overwrite_output_dir
```

### Analysing results

```bash
# Parse raw .out log files into structured JSON
python logs/folds/parse_folds.py --log_dir logs/folds/scratch

# Compute 95 % confidence intervals
python logs/folds/compute_ci_from_folds.py --log_dir logs/folds/scratch

# Paired t-test + TOST between the two strategies
python logs/folds/compare_model_logs_ttest.py \
  --log_dir_a logs/folds/pretrained \
  --log_dir_b logs/folds/scratch
```

---

## Data sources

| Dataset | Source | Species |
|---|---|---|
| Promoters | [EPDNew](https://epd.expasy.org/epd/) | Human / Mouse |
| Enhancers | [ENdb](http://www.licpathway.net/ENdb/) | Human / Mouse |
| Genome assembly | Ensembl GRCh38 / GRCm39 | Human / Mouse |
| CDS annotations | Ensembl GTF (protein-coding genes) | Human |

Sequences are stored in `scripts/datasets/` as tab-separated `.dat` files:

```
chr1    1000    3000    GENE_ID    +    ACGT...
```

---

## Dependencies

```
torch==1.13.1
transformers==4.29.2
accelerate==0.20.3
scikit-learn==1.2.2
evaluate==0.4.0
einops==0.6.1
peft==0.3.0
omegaconf==2.3.0
```

The scripts assume 4× GPU for training (A100 or RTX A5000). Single-GPU runs work by dropping `torchrun` and running `python` directly, with `--nproc_per_node=1` or by removing the DDP flags.

## Report

The [report](report/report.pdf) is in `report/report.pdf` and the source LaTeX files are in `report/`.
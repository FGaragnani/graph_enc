import os
import random
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForLanguageModeling

from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class ChromosomeDataset(TorchDataset):

    def __init__(
        self,
        data_path: str,
        only_protein_coding: bool = True,
        non_cds_sample_prob: float = 0.5,
        item_length_proportion: Optional[float] = None,
    ):
        """
        Initialize the ChromosomeDataset with paths to FASTA and GTF files.

        Args:
            data_path (str): Path to the directory containing FASTA and GTF files.
            only_protein_coding (bool): Whether to include only protein-coding genes. Defaults to True.
        """
        self.data_path: str = data_path
        self.only_protein_coding: bool = only_protein_coding
        self.non_cds_sample_prob: float = non_cds_sample_prob
        self.item_length_proportion: float = item_length_proportion if item_length_proportion is not None else 0.5

        if os.path.isdir(self.data_path):
            self.sequences, self.cds_annotations = self._load_from_folder(self.data_path)
        else:
            raise ValueError(f"Provided data_path '{self.data_path}' is not a directory containing FASTA and GTF files.")
        self.cds_annotations = self._collapse_gene_annotations(self.cds_annotations)
        self.dataset = self._create_dataset(self.cds_annotations)

    def _load_fasta_file(self, fasta_path: str) -> str:
        """
        Loads into memory the sequences from the FASTA file.
        
        Returns:
            str: The nucleotide sequences as a single concatenated string.
        """
        sequences = []
        with open(fasta_path, 'r') as fasta_file:
            sequence = ""
            for line in fasta_file:
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
        return "".join(sequences)
    
    def _load_gtf_file(self, gtf_path: str, offset: int = 0) -> List[dict]:

        transcripts: Dict[str, Dict] = defaultdict(lambda: {
            "strand": "",
            "cds_coords": [],
            "gene_id": None,
        })

        with open(gtf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] != "CDS":
                    continue

                start = int(fields[3]) - 1 + offset  # Convert to 0-based index
                end = int(fields[4]) + offset  # End is exclusive in Python slicing
                strand = fields[6]
                attributes = fields[8]

                attrs = {}
                for item in attributes.strip().strip(";").split(";"):
                    item = item.strip()
                    if not item:
                        continue
                    if " " not in item:
                        attrs[item] = ""
                        continue
                    key, value = item.split(" ", 1)
                    attrs[key] = value.strip().strip('"')
                transcript_id = attrs.get("transcript_id")
                gene_id = attrs.get("gene_id")
                if transcript_id is None:
                    print(f"Warning: No transcript_id found in attributes: {attributes}")
                    continue
                if self.only_protein_coding:
                    if attrs.get("gene_biotype") != "protein_coding":
                        continue

                entry = transcripts[transcript_id]
                entry["strand"] = strand
                if entry.get("gene_id") is None:
                    entry["gene_id"] = gene_id
                entry["cds_coords"].append((start, end))

        dataset = []
        for transcript_id, data in transcripts.items():
            if not data["cds_coords"]:
                raise ValueError(f"No CDS coordinates found for transcript {transcript_id}")
            coords = data["cds_coords"]
            coords.sort(key=lambda x: x[0])

            if data["strand"] == "-":
                coords = coords[::-1]
            
            dataset.append({
                "transcript_id": transcript_id,
                "gene_id": data.get("gene_id"),
                "strand": data["strand"],
                "cds_coords": coords
            })
        
        return dataset

    def _collapse_gene_annotations(self, annotations: List[dict]) -> dict[str, dict]:
        gene_annotations: Dict[str, dict] = {}
        for ann in annotations:
            gene_id = ann["gene_id"]
            if gene_id not in gene_annotations:
                gene_annotations[gene_id] = {
                    "strand": ann["strand"],
                    "cds_coords": []
                }
            gene_annotations[gene_id]["cds_coords"].extend(ann["cds_coords"])

        for gene_id, data in gene_annotations.items():
            data["cds_coords"] = self._merge_intervals(data["cds_coords"])

            gene_annotations[gene_id]["sequence"] = [
                self._get_indexed_sequence(start, end, reverse=False)
                for start, end in data["cds_coords"]
            ]

            new_coords: List[Tuple[int, int]] = []
            for seq in gene_annotations[gene_id]["cds_coords"]:
                start, end = seq
                if not new_coords:
                    new_coords.append((0, end - start))
                else:
                    new_coords.append((new_coords[-1][1], new_coords[-1][1] + (end - start)))
            gene_annotations[gene_id]["cds_coords"] = new_coords

        return {
            gene_id: {
                "strand": data["strand"],
                "cds_coords": data["cds_coords"],
                "sequence": data["sequence"]
            } for gene_id, data in gene_annotations.items()
        }
    
    def _create_dataset(self, annotations: dict[str, dict]) -> List[dict]:
        dataset: List[dict] = []
        for gene_id, entry in annotations.items():
            cds_coords = entry["cds_coords"]
            for seq in cds_coords:
                start, end = seq
                cds_len = end - start
                num_windows = int(((cds_len * self.item_length_proportion) + 1) // 2)
                for i in range(num_windows):
                    dataset.append({
                        "gene_id": gene_id,
                        "cds_coord": seq,
                        "sliding_window_id": i
                    })
        return dataset

    def _load_from_folder(self, root_dir: str) -> Tuple[str, List[dict]]:
        sequences = []
        annotations = []
        offset = 0

        subdirs = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        for subdir in sorted(subdirs):
            fasta_file, gtf_file = self._find_fasta_gtf_files(subdir)
            seq = self._load_fasta_file(fasta_file)
            ann = self._load_gtf_file(gtf_file, offset=offset)
            sequences.append(seq)
            annotations.extend(ann)

            offset += len(seq)

        return "".join(sequences), annotations
    
    def _length_cds_coord(self, cds_coords: list) -> int:
        return sum(end - start for start, end in cds_coords)

    def _find_fasta_gtf_files(self, folder: str) -> Tuple[str, str]:
        fasta_exts = (".fa", ".fasta", ".fna")
        fasta_files = []
        gtf_files = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            lower = name.lower()
            if lower.endswith(fasta_exts):
                fasta_files.append(path)
            elif lower.endswith(".gtf"):
                gtf_files.append(path)

        if not fasta_files:
            raise FileNotFoundError(f"No FASTA file found in {folder}")
        if not gtf_files:
            raise FileNotFoundError(f"No GTF file found in {folder}")

        return sorted(fasta_files)[0], sorted(gtf_files)[0]

    def _get_indexed_sequence(self, st_idx: int, end_idx: int, reverse: bool) -> str:
        """
        Retrieves a subsequence from the loaded sequences based on start and end indices.

        Args:
            st_idx (int): Start index of the subsequence.
            end_idx (int): End index of the subsequence.
            reverse (bool): Whether to return the reverse complement of the subsequence.

        Returns:
            str: The subsequence from st_idx to end_idx.
        """
        subsequence = self.sequences[st_idx:end_idx]
        if reverse:
            complement = str.maketrans('ACGT', 'TGCA')
            subsequence = subsequence.translate(complement)[::-1]
        return subsequence

    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
    
    def __len__(self) -> int:
        """
        Returns the total number of transcripts in the dataset.

        Returns:
            int: Length of the nucleotide sequences.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[str, List[int]]:
        """
        Retrieves the CDS sequence for the given index.

        Args:
            idx (int): Index of the CDS annotation to retrieve.
        Returns:
            Tuple[str, List[int]]: The CDS nucleotide sequence and a mask indicating CDS regions.
        """

        ann = self.dataset[idx]

        entry = ann["cds_coord"]
        sliding_window_id = ann["sliding_window_id"]
        gene_id = ann["gene_id"]
        
        strand = self.cds_annotations[gene_id]["strand"]
        sequence = self.cds_annotations[gene_id]["sequence"]

        #    [CCC MMM CCC]
        #         3,5

        # 0: [CCC MMM C]
        # 1: [ CC MMM CC]
        # 2: [  C MMM CCC]
        cds_start, cds_end = entry
        cds_len = cds_end - cds_start

        flank_left = int(cds_len * self.item_length_proportion) - sliding_window_id
        flank_right = int(cds_len * self.item_length_proportion // 2) + sliding_window_id
        # window_len = int((cds_end - cds_start) + (flank_left) + (flank_right))

        region_start = int(max(0, cds_start - flank_left))
        region_end = int(min(len(sequence), cds_end + flank_right))
        dna_sequence = sequence[region_start:region_end]
        mask = [0] * (region_end - region_start)
        for i in range(max(cds_start, region_start), min(cds_end, region_end)):
            mask[i - region_start] = 1

        if strand == "-":
            complement = str.maketrans('ACGT', 'TGCA')
            dna_sequence = dna_sequence.translate(complement)[::-1]
            mask = mask[::-1]

        return dna_sequence, mask
    
class CDSMaskingDataset(TorchDataset):
    def __init__(self, base_dataset: ChromosomeDataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sequence, cds_mask = self.base_dataset[idx]
        return {"text": sequence, "cds_mask": cds_mask}


class DataCollatorForCDSMaskedLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        mlm_probability=0.15,
        non_cds_mlm_probability=0.03,
        pad_to_multiple_of=None,
        max_length=None,
        pad_to_max_length=False,
    ):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of)
        self.non_cds_mlm_probability = non_cds_mlm_probability
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, examples):
        if not examples:
            return {}
        if "text" in examples[0]:
            texts = [ex["text"] for ex in examples]
            cds_masks = [ex["cds_mask"] for ex in examples]
            padding = "max_length" if self.pad_to_max_length and self.max_length is not None else True
            batch = self.tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )

            offset_mappings = batch.pop("offset_mapping")
            token_cds_masks = []
            for cds_mask, offsets in zip(cds_masks, offset_mappings):
                token_mask = []
                for start, end in offsets:
                    if start == end:
                        token_mask.append(0)
                    else:
                        token_mask.append(1 if any(cds_mask[start:end]) else 0)
                token_cds_masks.append(token_mask)

            batch["cds_mask"] = token_cds_masks
            batch = {k: torch.tensor(v) for k, v in batch.items()}
        else:
            try:
                batch = self.tokenizer.pad(
                    examples,
                    padding=True,
                    return_special_tokens_mask=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )
            except TypeError:
                batch = self.tokenizer.pad(
                    examples,
                    padding=True,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )
                special_masks = []
                for input_ids in batch["input_ids"]:
                    special_masks.append(
                        self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
                    )
                batch["special_tokens_mask"] = special_masks

        if self.tokenizer.mask_token is None:
            raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling.")

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        special_tokens_mask = batch.pop("special_tokens_mask")
        if not torch.is_tensor(special_tokens_mask):
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        cds_mask = batch.pop("cds_mask", None)
        if cds_mask is not None:
            if not torch.is_tensor(cds_mask):
                cds_mask = torch.tensor(cds_mask, dtype=torch.float)
            else:
                cds_mask = cds_mask.float()

        probability_matrix = torch.full(labels.shape, self.non_cds_mlm_probability, device=labels.device)
        if cds_mask is not None:
            cds_probability = torch.full(labels.shape, self.mlm_probability, device=labels.device)
            probability_matrix = torch.where(cds_mask > 0.0, cds_probability, probability_matrix)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        input_ids[indices_random] = random_words[indices_random]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch
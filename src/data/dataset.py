import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForLanguageModeling

from collections import defaultdict
from typing import List, Dict, Tuple

class ChromosomeDataset(TorchDataset):

    def __init__(self, fasta_path: str, gtf_path: str, only_protein_coding: bool = True):
        """
        Initialize the ChromosomeDataset with paths to FASTA and GTF files.

        Args:
            fasta_path (str): Path to the FASTA file containing chromosome sequences.
            gtf_path (str): Path to the GTF file containing gene annotations.
            only_protein_coding (bool): Whether to include only protein-coding genes. Defaults to True.
        """
        self.fasta_path = fasta_path
        self.gtf_path = gtf_path
        self.only_protein_coding = only_protein_coding
        self.sequences = self._load_fasta()
        self.cds_annotations = self._load_gtf()

    def _load_fasta(self) -> str:
        """
        Loads into memory the sequences from the FASTA file.
        
        Returns:
            str: The nucleotide sequences as a single concatenated string.
        """
        sequences = []
        with open(self.fasta_path, 'r') as fasta_file:
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
    
    def _load_gtf(self) -> List[dict]:

        transcripts: Dict[str, Dict] = defaultdict(lambda: {
            "strand": "",
            "cds_coords": [],
            "gene_id": None,
        })

        with open(self.gtf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] != "CDS":
                    continue

                start = int(fields[3]) - 1  # Convert to 0-based index
                end = int(fields[4])  # End is exclusive in Python slicing
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
    
    def __len__(self) -> int:
        """
        Returns the total number of transcripts in the dataset.

        Returns:
            int: Length of the nucleotide sequences.
        """
        return len(self.cds_annotations)

    def __getitem__(self, idx: int) -> Tuple[str, List[int]]:
        """
        Retrieves the CDS sequence for the given index.

        Args:
            idx (int): Index of the CDS annotation to retrieve.
        Returns:
            Tuple[str, List[int]]: The CDS nucleotide sequence and a mask indicating CDS regions.
        """

        entry = self.cds_annotations[idx]
        strand = entry["strand"]

        cds_coords = entry["cds_coords"]
        cds_len = sum(end - start for start, end in cds_coords)

        cds_start = min(start for start, _ in cds_coords)
        cds_end = max(end for _, end in cds_coords)

        flank_len = cds_len // 2
        region_start = max(0, cds_start - flank_len)
        region_end = min(len(self.sequences), cds_end + flank_len)

        dna_sequence = self.sequences[region_start:region_end]
        mask = [0] * (region_end - region_start)
        for start, end in cds_coords:
            for i in range(max(start, region_start), min(end, region_end)):
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
        pad_to_multiple_of=None,
        max_length=None,
        pad_to_max_length=False,
    ):
        super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability, pad_to_multiple_of=pad_to_multiple_of)
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

        cds_mask = batch.pop("cds_mask")
        if not torch.is_tensor(cds_mask):
            cds_mask = torch.tensor(cds_mask, dtype=torch.float)
        else:
            cds_mask = cds_mask.float()

        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=labels.device)
        probability_matrix = probability_matrix * cds_mask
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
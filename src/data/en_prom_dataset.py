import os
from enum import Enum
from typing import List, Optional, Dict

from torch.utils.data import Dataset as TorchDataset

class ItemType(Enum):
    PROMOTER = 0
    ENHANCER = 1

class PEDatasetItem:
    def __init__(self, sequence_init: int, sequence_end: int, chr_idx: int, type: ItemType, sequence: Optional[str] = None):
        self.sequence_init: int = sequence_init
        self.sequence_end: int = sequence_end
        self.chr_idx: int = chr_idx
        self.type: ItemType = type
        self.sequence: Optional[str] = sequence

    def is_promoter(self) -> bool:
        return self.type == ItemType.PROMOTER
    
    def is_enhancer(self) -> bool:
        return not self.is_promoter()
    
    def get_chr_idx(self) -> int:
        return self.chr_idx
    
    def get_sequence(self, chr_seq: Optional[str] = None) -> str:
        """
            Get the sequence of this item. If the sequence is not already loaded, it will be loaded from the provided chromosome sequence.
            Practically: for enhancers, it has to be computed; for promoters, it is already loaded in the dataset.
        """
        if self.type == ItemType.PROMOTER:
            if self.sequence is None:
                raise ValueError("Promoter sequence must be provided in the dataset.")
            return self.sequence
        
        elif self.type == ItemType.ENHANCER:
            if chr_seq is None:
                raise ValueError("Chromosome sequence must be provided to compute the sequence of this item.")
            return chr_seq[self.sequence_init:self.sequence_end]
        
        raise ValueError("Invalid item type.")

class PromoterEnhancerDataset(TorchDataset):

    def __init__(self, dir: str = "scripts/datasets", genome_data_path: Optional[str] = None):
        """
            Build a Promoter / Enhancer Dataset by passing a directory containing 'enhancers.dat' and 'promoters.dat' files.
        """
        super().__init__()
        self.dir = dir
        self.genome_data_path = genome_data_path
        self.chromosome_sequences = self._load_chromosome_sequences() if self.genome_data_path is not None else {}
        self.data: List[PEDatasetItem] = self._load_data()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        item = self.data[idx]
        if item.is_promoter():
            return item.get_sequence()

        chr_seq = self.chromosome_sequences.get(item.get_chr_idx())
        if chr_seq is None:
            loaded = sorted(self.chromosome_sequences.keys())
            raise ValueError(
                "Genome sequence not loaded for chromosome index "
                f"{item.get_chr_idx()} (dataset index {idx}). "
                f"Loaded chromosome indices: {loaded}. "
                "Pass genome_data_path with chrN/ subfolders when creating PromoterEnhancerDataset."
            )
        return item.get_sequence(chr_seq)

    def _load_data(self) -> List[PEDatasetItem]:
        data: List[PEDatasetItem] = []
        data.extend(self._load_enhancers())
        data.extend(self._load_promoters())
        return data
    
    def _load_promoters(self) -> List[PEDatasetItem]:
        promoters: List[PEDatasetItem] = []
        current_sequence_parts: List[str] = []

        with open(f"{self.dir}/promoters.dat", "r") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                if line.startswith(">"):
                    if current_sequence_parts:
                        promoters.append(
                            PEDatasetItem(0, 0, -1, ItemType.PROMOTER, sequence="".join(current_sequence_parts))
                        )
                        current_sequence_parts = []
                    continue

                current_sequence_parts.append(line)

        if current_sequence_parts:
            promoters.append(PEDatasetItem(0, 0, -1, ItemType.PROMOTER, sequence="".join(current_sequence_parts)))

        return promoters
    
    def _load_enhancers(self) -> List[PEDatasetItem]:
        enhancers: List[PEDatasetItem] = []
        with open(f"{self.dir}/enhancers.dat", "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                cols = line.split("\t")
                if len(cols) < 4 or cols[0] == "Pubmed":
                    continue

                # chr21 for instance -> 21
                chr_name = cols[1].strip()
                if chr_name.startswith("chr"):
                    chr_name = chr_name[3:]
                if not chr_name.isdigit():
                    continue
                chr_idx = int(chr_name)

                try:
                    sequence_init = int(cols[2])
                    sequence_end = int(cols[3])
                except ValueError:
                    continue

                enhancers.append(PEDatasetItem(sequence_init, sequence_end, chr_idx, ItemType.ENHANCER))
        unique_chr_idxs = sorted({item.get_chr_idx() for item in enhancers})
        print(f"[PE DEBUG] Loaded {len(enhancers)} enhancers. Referenced chromosome indices: {unique_chr_idxs}")
        return enhancers

    def _load_chromosome_sequences(self) -> Dict[int, str]:
        if self.genome_data_path is None:
            print("No genome_data_path provided, chromosome sequences will not be loaded. Enhancer sequences will not be available.")
            return {}

        if not os.path.isdir(self.genome_data_path):
            raise ValueError(f"Provided genome_data_path '{self.genome_data_path}' is not a directory.")

        chromosome_sequences: Dict[int, str] = {}

        print(f"[PE DEBUG] genome_data_path={self.genome_data_path}")

        subdirs = [
            os.path.join(self.genome_data_path, name)
            for name in os.listdir(self.genome_data_path)
            if os.path.isdir(os.path.join(self.genome_data_path, name))
        ]
        print(f"[PE DEBUG] Found {len(subdirs)} immediate subdirectories under genome_data_path")

        for subdir in sorted(subdirs):
            chromosome_name = os.path.basename(subdir)
            if not chromosome_name.startswith("chr"):
                print(f"[PE DEBUG] Skipping folder '{chromosome_name}': does not start with 'chr'")
                continue

            numeric_part = chromosome_name[3:]
            if not numeric_part.isdigit():
                print(f"[PE DEBUG] Skipping folder '{chromosome_name}': suffix is not numeric")
                continue

            fasta_file = self._find_fasta_file(subdir)
            if fasta_file is None:
                print(f"[PE DEBUG] Skipping folder '{chromosome_name}': no FASTA file found")
                continue

            chromosome_sequence = self._load_fasta_file(fasta_file)
            chromosome_idx = int(numeric_part)

            chromosome_sequences[chromosome_idx] = chromosome_sequence
            print(
                f"[PE DEBUG] Loaded {chromosome_name} -> index {chromosome_idx} "
                f"from {fasta_file} (length={len(chromosome_sequence)})"
            )
        print(f"[PE DEBUG] Loaded chromosome sequences for indices: {sorted(chromosome_sequences.keys())}")

        return chromosome_sequences

    def _find_fasta_file(self, folder: str) -> Optional[str]:
        fasta_exts = (".fa", ".fasta", ".fna")
        fasta_files = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(fasta_exts):
                fasta_files.append(path)
        if not fasta_files:
            print(f"No FASTA file found in {folder}, skipping.")
            return None
        return sorted(fasta_files)[0]

    def _load_fasta_file(self, fasta_path: str) -> str:
        sequences = []
        with open(fasta_path, "r") as fasta_file:
            sequence = ""
            for line in fasta_file:
                if line.startswith(">"):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ""
                else:
                    sequence += line.strip()
            if sequence:
                sequences.append(sequence)
        return "".join(sequences)
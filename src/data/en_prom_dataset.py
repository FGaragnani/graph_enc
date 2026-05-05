import os
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset as TorchDataset

class ItemType(Enum):
    PROMOTER = 0
    ENHANCER = 1

class PEDatasetItem:
    def __init__(
        self,
        sequence_init: int,
        sequence_end: int,
        chr_idx: int,
        type: ItemType,
        strand: Optional[str] = None,
        sequence: Optional[str] = None,
        promoter_window_size: Optional[int] = None,
    ):
        self.sequence_init: int = sequence_init
        self.sequence_end: int = sequence_end
        self.chr_idx: int = chr_idx
        self.type: ItemType = type
        self.strand: Optional[str] = strand
        self.sequence: Optional[str] = sequence
        self.promoter_window_size: Optional[int] = promoter_window_size

    def is_promoter(self) -> bool:
        return self.type == ItemType.PROMOTER
    
    def is_enhancer(self) -> bool:
        return not self.is_promoter()
    
    def get_chr_idx(self) -> int:
        return self.chr_idx

    def get_start(self) -> int:
        return self.sequence_init

    def get_end(self) -> int:
        return self.sequence_end

    def get_strand(self) -> Optional[str]:
        return self.strand
    
    def len(self) -> int:
        if self.is_promoter():
            if self.sequence is not None:
                return len(self.sequence)
            if self.promoter_window_size is not None:
                return self.promoter_window_size
            return max(0, self.sequence_end - self.sequence_init)
        return self.sequence_end - self.sequence_init
    
    def _reverse_complement(self, sequence: str) -> str:
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        return sequence.translate(complement)[::-1]

    def _get_promoter_bounds(self, chr_seq: str, window_size: int) -> Tuple[int, int]:
        center = (self.sequence_init + self.sequence_end) // 2
        half_window = window_size // 2
        window_start = max(0, center - half_window)
        window_end = window_start + window_size

        if window_end > len(chr_seq):
            window_end = len(chr_seq)
            window_start = max(0, window_end - window_size)

        return window_start, window_end

    def get_sequence(self, chr_seq: Optional[str] = None, promoter_window_size: Optional[int] = None) -> str:
        """
            Get the sequence of this item. If the sequence is not already loaded, it will be loaded from the provided chromosome sequence.
            Practically: for enhancers, it has to be computed; for promoters, it is already loaded in the dataset.
        """
        if self.type == ItemType.PROMOTER:
            if self.sequence is None:
                if chr_seq is None:
                    raise ValueError("Chromosome sequence must be provided to compute the sequence of this item.")
                window_size = promoter_window_size or self.promoter_window_size or max(0, self.sequence_end - self.sequence_init)
                if window_size <= 0:
                    raise ValueError("Promoter window size must be positive.")
                window_start, window_end = self._get_promoter_bounds(chr_seq, window_size)
                sequence = chr_seq[window_start:window_end]
                if self.strand == "-":
                    sequence = self._reverse_complement(sequence)
                return sequence
            return self.sequence
        
        elif self.type == ItemType.ENHANCER:
            if chr_seq is None:
                raise ValueError("Chromosome sequence must be provided to compute the sequence of this item.")
            return chr_seq[self.sequence_init:self.sequence_end]
        
        raise ValueError("Invalid item type.")

class PromoterEnhancerDataset(TorchDataset):

    def __init__(
        self,
        dir: str = "scripts/datasets",
        genome_data_path: Optional[str] = None,
        target_enhancer_fraction: Optional[float] = None,
        balance_seed: int = 42,
        apply_rebalancing: bool = True,
        promoter_window_size: int = 2000,
    ):
        """
            Build a Promoter / Enhancer Dataset by passing a directory containing 'enhancers.dat' and 'promoters.dat' files.
        """
        super().__init__()
        self.dir = dir
        self.genome_data_path = genome_data_path
        self.target_enhancer_fraction = target_enhancer_fraction
        self.balance_seed = balance_seed
        self.promoter_window_size = promoter_window_size
        self.chromosome_sequences = self._load_chromosome_sequences() if self.genome_data_path is not None else {}
        self.data: List[PEDatasetItem] = self._load_data()
        if apply_rebalancing:
            self.data = self._rebalance_data(self.data)
            self._print_class_distribution(self.data, prefix="Final")
        else:
            self._print_class_distribution(self.data, prefix="Loaded")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        item = self.data[idx]
        chr_seq = self.chromosome_sequences.get(item.get_chr_idx())
        if chr_seq is None:
            raise ValueError(
                "Genome sequence not loaded for chromosome index "
                f"{item.get_chr_idx()}. Pass genome_data_path when creating PromoterEnhancerDataset."
            )
        return item.get_sequence(chr_seq, self.promoter_window_size)

    def _load_data(self) -> List[PEDatasetItem]:
        data: List[PEDatasetItem] = []
        data.extend(self._load_enhancers())
        data.extend(self._load_promoters())
        self._print_class_distribution(data, prefix="Loaded")
        return data

    def _print_class_distribution(self, data: List[PEDatasetItem], prefix: str) -> None:
        num_enhancers = sum(1 for item in data if item.is_enhancer())
        num_promoters = len(data) - num_enhancers
        total = len(data)
        enh_fraction = (num_enhancers / total) if total > 0 else 0.0
        print(
            f"{prefix} PE dataset: total={total}, enhancers={num_enhancers}, "
            f"promoters={num_promoters}, enhancer_fraction={enh_fraction:.4f}"
        )
        
        lengths_enhancers = [item.len() for item in data if item.is_enhancer()]
        lengths_promoters = [item.len() for item in data if item.is_promoter()]
        avg_enhancer_length = sum(lengths_enhancers) / len(lengths_enhancers) if lengths_enhancers else 0
        avg_promoter_length = sum(lengths_promoters) / len(lengths_promoters) if lengths_promoters else 0
        std_enhancer_length = (sum((l - avg_enhancer_length) ** 2 for l in lengths_enhancers) / len(lengths_enhancers)) ** 0.5 if lengths_enhancers else 0
        std_promoter_length = (sum((l - avg_promoter_length) ** 2 for l in lengths_promoters) / len(lengths_promoters)) ** 0.5 if lengths_promoters else 0
        print(
            f"{prefix} PE dataset: average enhancer length={avg_enhancer_length:.2f}, "
            f"average promoter length={avg_promoter_length:.2f}, "
            f"std enhancer length={std_enhancer_length:.2f}, "
            f"std promoter length={std_promoter_length:.2f}"
        )

    def _rebalance_data(self, data: List[PEDatasetItem]) -> List[PEDatasetItem]:
        if self.target_enhancer_fraction is None:
            return data

        target = self.target_enhancer_fraction
        enhancers = [item for item in data if item.is_enhancer()]
        promoters = [item for item in data if item.is_promoter()]

        if not enhancers or not promoters:
            print("Skipping rebalancing because one class is empty.")
            return data

        rng = random.Random(self.balance_seed)
        desired_enhancers = int(round((target / (1.0 - target)) * len(promoters)))

        if desired_enhancers <= len(enhancers):
            selected_enhancers = rng.sample(enhancers, desired_enhancers)
            selected_promoters = promoters
            dropped = len(enhancers) - len(selected_enhancers)
            print(
                f"Rebalancing PE dataset to enhancer_fraction~{target:.3f}: "
                f"downsampled enhancers by {dropped} samples."
            )
        else:
            desired_promoters = int(round(((1.0 - target) / target) * len(enhancers)))
            selected_enhancers = enhancers
            selected_promoters = rng.sample(promoters, desired_promoters)
            dropped = len(promoters) - len(selected_promoters)
            print(
                f"Rebalancing PE dataset to enhancer_fraction~{target:.3f}: "
                f"downsampled promoters by {dropped} samples."
            )

        balanced = selected_enhancers + selected_promoters
        rng.shuffle(balanced)
        return balanced
    
    def _load_promoters(self) -> List[PEDatasetItem]:
        promoters: List[PEDatasetItem] = []

        with open(f"{self.dir}/promoters.dat", "r") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                cols = line.split()
                if len(cols) < 6:
                    continue

                chr_name = cols[0].strip()
                if chr_name.startswith("chr"):
                    chr_name = chr_name[3:]
                if not chr_name.isdigit():
                    continue
                chr_idx = int(chr_name)

                try:
                    sequence_init = int(cols[1])
                    sequence_end = int(cols[2])
                except ValueError:
                    continue

                strand = cols[5].strip()
                if strand not in {"+", "-"}:
                    continue

                promoters.append(
                    PEDatasetItem(
                        sequence_init,
                        sequence_end,
                        chr_idx,
                        ItemType.PROMOTER,
                        strand=strand,
                        promoter_window_size=self.promoter_window_size,
                    )
                )

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
        return enhancers

    def _load_chromosome_sequences(self) -> Dict[int, str]:
        if self.genome_data_path is None:
            print("No genome_data_path provided, chromosome sequences will not be loaded. Enhancer sequences will not be available.")
            return {}

        if not os.path.isdir(self.genome_data_path):
            raise ValueError(f"Provided genome_data_path '{self.genome_data_path}' is not a directory.")

        chromosome_sequences: Dict[int, str] = {}

        subdirs = [
            os.path.join(self.genome_data_path, name)
            for name in os.listdir(self.genome_data_path)
            if os.path.isdir(os.path.join(self.genome_data_path, name))
        ]
        for subdir in sorted(subdirs):
            fasta_file = self._find_fasta_file(subdir)
            if fasta_file is None:
                continue
            chromosome_sequence = self._load_fasta_file(fasta_file)

            chromosome_name = os.path.basename(subdir)
            if chromosome_name.startswith("human_chr_"):
                chromosome_name = chromosome_name[len("human_chr_"):]
            elif chromosome_name.startswith("human_chr"):
                chromosome_name = chromosome_name[len("human_chr"):]
            elif chromosome_name.startswith("chr"):
                chromosome_name = chromosome_name[3:]

            if not chromosome_name.isdigit():
                continue
            chromosome_idx = int(chromosome_name)

            chromosome_sequences[chromosome_idx] = chromosome_sequence
        print(f"Loaded chromosome sequences for indices: {list(chromosome_sequences.keys())}")

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
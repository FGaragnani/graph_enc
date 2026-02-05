from torch.utils.data import Dataset

from collections import defaultdict
from typing import List, Dict, Tuple

class ChromosomeDataset(Dataset):

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
            "cds_coords": []
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

                attrs = dict(
                    item.strip().replace('"', '').split(' ')
                    for item in attributes.strip(';').split('; ')
                )
                transcript_id = attrs.get("transcript_id")
                if transcript_id is None:
                    print(f"Warning: No transcript_id found in attributes: {attributes}")
                    continue
                if self.only_protein_coding:
                    if attrs.get("gene_biotype") != "protein_coding":
                        continue

                entry = transcripts[transcript_id]
                entry["strand"] = strand
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
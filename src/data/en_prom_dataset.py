from enum import Enum

from torch.utils.data import Dataset as TorchDataset

class ItemType(Enum):
    PROMOTER = 0
    ENHANCER = 1

class PromoterEnhancerDataset(TorchDataset):

    class PEDatasetItem:
        def __init__(self, sequence_init: int, sequence_end: int, chr_idx: int, type: ItemType):
            self.sequence_init: int = sequence_init
            self.sequence_end: int = sequence_end
            self.chr_idx: int = chr_idx
            self.type: ItemType = type

        def is_promoter(self) -> bool:
            return self.type == ItemType.PROMOTER
        
        def is_enhancer(self) -> bool:
            return not self.is_promoter()
        
        def get_chr_idx(self) -> int:
            return self.chr_idx
        
        def get_sequence(self) -> tuple[int, int]:
            return (self.sequence_init, self.sequence_end)

    def __init__(self, dir: str = "scripts/datasets"):
        """
            Build a Promoter / Enhancer Dataset by passing a directory containing 'enhancers.dat' and 'promoters.dat' files.
        """
        super().__init__()
        self.dir = dir
        self.data = self._load_data()

    def _load_data(self):
        data = []
        return data
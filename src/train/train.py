from graph_enc.src.data.dataset import ChromosomeDataset
from graph_enc.src.model.model import BertForMaskedLM
from graph_enc.src.model.run_mlm import main as run_mlm

if __name__ == "__main__":
    run_mlm()
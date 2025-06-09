from datasets import Dataset

def load_pairs_tsv(path: str):
    """Yields dicts {'source': List[str], 'target': List[str]} from TSV."""
    with open(path) as f:
        for line in f:
            src, tgt = line.strip().split("\t")
            yield {"source": list(src), "target": list(tgt)}

def build_datasets(train_path: str, valid_path: str):
    raw_train = list(load_pairs_tsv(train_path))
    raw_valid = list(load_pairs_tsv(valid_path))
    return Dataset.from_list(raw_train), Dataset.from_list(raw_valid)


from datasets import Dataset


def load_pairs_tsv(path: str):
    """Carica un TSV con colonne RNA_sequence_x e RNA_sequence_y e restituisce dicts {'source': List[str], 'target': List[str]}."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = row.get("RNA_sequence_x")
            tgt = row.get("RNA_sequence_y")
            if not src or not tgt:
                continue  # salta righe incomplete
            yield {"source": list(src.strip()), "target": list(tgt.strip())}

def build_datasets(train_path: str, valid_path: str):
    raw_train = list(load_pairs_tsv(train_path))
    raw_valid = list(load_pairs_tsv(valid_path))
    return Dataset.from_list(raw_train), Dataset.from_list(raw_valid)

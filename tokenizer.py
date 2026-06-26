from transformers import PreTrainedTokenizer

class NucTokenizer(PreTrainedTokenizer):
    """
    Tokenizer minimale per RNA con tokenizzazione 1-mer e simboli speciali.
    """
    def __init__(self):
        vocab = ["PAD", "EOS", "UNK", "A", "U", "C", "G", "BOS"]
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        super().__init__(
            pad_token="PAD",
            eos_token="EOS",
            unk_token="UNK",
            bos_token="BOS",
            pad_token_id=self.vocab["PAD"],
            eos_token_id=self.vocab["EOS"],
            unk_token_id=self.vocab["UNK"],
            bos_token_id=self.vocab["BOS"],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def added_tokens_encoder(self):
        return {}

    def _tokenize(self, text: str):
        return list(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab["UNK"])

    def _convert_id_to_token(self, index: int):
        return self.id_to_token.get(index, "UNK")

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        return []

    def save_pretrained(self, save_directory: str, **kwargs):
        return []


# === Funzioni di supporto ===
tokenizer = NucTokenizer()

def tokenize_batch(batch, max_length=1024):
    if "source" in batch and "target" in batch:
        src_key = "source"
        tgt_key = "target"

    elif "seq1" in batch and "seq2" in batch:
        src_key = "seq1"
        tgt_key = "seq2"
    
    elif "RNA_sequence_x" in batch and "RNA_sequence_y" in batch:
        src_key = "RNA_sequence_x"
        tgt_key = "RNA_sequence_y"
    
    else:
        raise KeyError(
            f"Colonne non riconosciute. Trovate: {list(batch.keys())}. "
            "Attese: source/target, seq1/seq2 oppure RNA_sequence_x/RNA_sequence_y."
        )

    src_raw_texts = ["".join(x) if isinstance(x, list) else x for x in batch[src_key]]
    tgt_raw_texts = ["".join(x) if isinstance(x, list) else x for x in batch[tgt_key]]

    bos_id = tokenizer.vocab["BOS"]
    eos_id = tokenizer.vocab["EOS"]
    unk_id = tokenizer.vocab["UNK"]

    # PADDING DINAMICO: qui NON si imbottisce a lunghezza fissa. Si restituiscono
    # sequenze a lunghezza variabile (liste di id); il padding fino alla sequenza piu'
    # lunga del singolo batch viene fatto dal data_collator in fase di training.
    # Gli id sono costruiti a mano (1-mer): cosi' BOS/EOS sono inseriti in modo
    # affidabile e non ci sono spazi che diventerebbero UNK.
    input_ids_list, attention_list, dec_list, lab_list = [], [], [], []
    for src_text, tgt_text in zip(src_raw_texts, tgt_raw_texts):
        src_text = str(src_text).strip()
        tgt_text = str(tgt_text).strip()

        # Source (RNA1): solo i nucleotidi, nessun token speciale.
        src_ids = [tokenizer.vocab.get(ch, unk_id) for ch in src_text][:max_length]

        # Target (RNA2): BOS + nucleotidi + EOS, cosi' il decoder impara a generare
        # anche il PRIMO nucleotide (da BOS), coerente con generate() che parte da BOS.
        tgt_ids = [bos_id] + [tokenizer.vocab.get(ch, unk_id) for ch in tgt_text] + [eos_id]
        tgt_ids = tgt_ids[:max_length]

        input_ids_list.append(src_ids)
        attention_list.append([1] * len(src_ids))
        # Teacher forcing: decoder_input = tutto tranne l'ultimo, labels = shiftato di 1.
        dec_list.append(tgt_ids[:-1])
        lab_list.append(tgt_ids[1:])

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_list,
        "decoder_input_ids": dec_list,
        "labels": lab_list,
    }


def detokenize_batch(batch_ids):
    seqs = []
    id2tok = tokenizer.id_to_token

    for ids in batch_ids:
        tokens = [id2tok.get(i, "") for i in ids if id2tok.get(i) not in ["PAD", "EOS"]]
        seqs.append("".join(tokens))
    return seqs

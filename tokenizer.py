from transformers import PreTrainedTokenizer
import torch

class NucTokenizer(PreTrainedTokenizer):
    """
    A bare‐bones tokenizer for A,U,C,G plus PAD/STOP.
    """

    def __init__(self):
        # our six symbols
        vocab = ["PAD", "STOP", "A", "U", "C", "G"]
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        super().__init__(
            pad_token="PAD",
            eos_token="STOP",
            unk_token="STOP",
            pad_token_id=self.vocab["PAD"],
            eos_token_id=self.vocab["STOP"],
            unk_token_id=self.vocab["STOP"],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def added_tokens_encoder(self):
        # safely return empty mapping
        return {}

    def _tokenize(self, text: str):
        # split on characters
        return list(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab["STOP"])

    def _convert_id_to_token(self, index: int):
        return self.id_to_token.get(index, "STOP")

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        # no files to save
        return []
        
    def save_pretrained(self, save_directory: str, **kwargs):
        # override to no-op, avoids base-class save logic
        return []

# instantiate a single global tokenizer
tokenizer = NucTokenizer()


def tokenize_batch(batch):
    src_key = "source" if "source" in batch else "seq1"
    tgt_key = "target" if "target" in batch else "seq2"
    
    # append eos to each string of tokens
    if isinstance(batch[src_key][0], list):
        src_texts = ["".join(x + [tokenizer.eos_token]) for x in batch[src_key]]
    else:
        src_texts = [x + tokenizer.eos_token for x in batch[src_key]]
           
    if isinstance(batch[tgt_key][0], list):
        tgt_texts = ["".join(x + [tokenizer.eos_token]) for x in batch[tgt_key]]
    else:
        tgt_texts = [x + tokenizer.eos_token for x in batch[tgt_key]]
    
    print("Tutti i token della prima sequenza target:", tokenizer.encode(tgt_texts[0]))
    print("Tutti i token della prima sequenza source:", tokenizer.encode(src_texts[0]))
    
    # use the base class encode_batch
    enc = tokenizer(src_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")   # max_length è stato modificato
    tgt_enc = tokenizer(tgt_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")   # max_length è stato modificato

    enc["labels"] = tgt_enc["input_ids"]
    
   
    return enc

    

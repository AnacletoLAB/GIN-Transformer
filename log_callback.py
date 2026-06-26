from transformers import TrainerCallback
import torch
from tokenizer import tokenize_batch, tokenizer

class GenerationLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        # Esempio fisso per il monitoraggio qualitativo
        seq1 = "AGCGACCGGGGGUUGGAUGGAUA"
        seq2 = "CUGGCGGUGUGGACAGGGCGUGA"

        batch = {"source": [seq1], "target": [seq2]}
        tokenized = tokenize_batch(batch)

        # tokenize_batch ora restituisce liste a lunghezza variabile: qui c'e' un solo
        # esempio, quindi si converte direttamente a tensore (niente padding necessario).
        input_ids = torch.tensor(tokenized["input_ids"], device=model.device)
        attention_mask = torch.tensor(tokenized["attention_mask"], device=model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print(f"\n--- Generazione di esempio (eval) ---\nseq1: {seq1}\nseq2 target: {seq2}\nseq2 generata: {decoded[0]}\n")

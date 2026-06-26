from transformers import TrainerCallback
import torch
from tokenizer import tokenize_batch, tokenizer  # usa il tokenizer globale e la tua funzione custom

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
            # print(f"ID token STOP: {tokenizer.vocab['STOP']}")
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
                # eos_token_id=tokenizer.vocab["STOP"],
                # early_stopping=True  # opzionale, ma utile
            )
        
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # print(f"Sequence of generated token ids: {generated_ids[0].tolist()}")

        # print("\n------ Generazione di esempio ------")
        # print(f"seq1: {seq1}")
        # print(f"seq2: {seq2}")
        # print(f"Predicted: {decoded[0]}")
        # print(f" --> Target atteso: {seq2}")
        # print("------------------------------------------------\n")

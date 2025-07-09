import os
import pandas as pd
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # comment this line for data parallel (train on all avaiables GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # comment this line for data parallel (train on all avaiables GPUs) #uso la seconda GPU
from datasets import Dataset
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from data import build_datasets, load_pairs_tsv
from tokenizer import tokenize_batch, tokenizer
from model import NucConfig, NucTransformer
from utils import set_seed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem, falling back to numerical ordering")

# batch_size: è la suddivisione dei dati che non vengono processati tutti insieme
# ma suddivisi in gruppi

# grad_acc_steps = gradient accumulation: simula un batch più grande 
# accumulando il gradiente ogni 2 batch prima dell’aggiornamento.

# warmup_steps = 1000: Nei primi 1000 step, il learning rate cresce da 0 → 3e-5 
# in modo lineare (o secondo lo scheduler che hai impostato: nel tuo caso, cosine)

def main(
    train_file="data/dry_run/train_final.tsv", valid_file="data/dry_run/validation_raw.tsv",
    output_dir="ckpt/dry_run", logging_dir="ckpt/dry_run/logs",
    learning_rate=3e-5, train_batch_size=32, eval_batch_size=32,
    num_epochs=20, bf16=True, grad_acc_steps=2, warmup_steps=1000,
    scheduler="cosine", early_stop=3, num_workers=8, seed=42 
    ):
    
    set_seed(seed)
    # data
    train_ds, valid_ds = build_datasets(train_file, valid_file)
    train_ds = train_ds.map(tokenize_batch, batched=True)
    valid_ds = valid_ds.map(tokenize_batch, batched=True)

    # model & config
    config = NucConfig(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, max_len=128) # QUI
    model  = NucTransformer(config)

    # training args
    # early_stop mi dice di quanto deve andare avanti il training 
    # dopo che il modello ha smesso di migliorare, qui early_stop = 3
    # cioè se non migliora più per 3 epoche consecutive si ferma il training
    # ed è il parametro early_stops = 3
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir, logging_dir=logging_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=learning_rate, num_train_epochs=num_epochs,
        bf16=bf16, warmup_steps=warmup_steps, lr_scheduler_type=scheduler,
        eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="epoch", report_to=["tensorboard"],
        dataloader_num_workers=num_workers, save_total_limit=3,
        # save_total_limit serve per salvare al massimo 3 checkpoint
        load_best_model_at_end=True,             # <-- load best on early stop, con il numero più piccolo di eval_loss
        metric_for_best_model="eval_loss",       # <-- which metric to monitor
        greater_is_better=False,                 # <-- lower loss is better
        # overwrite_output_dir serve per ricominciare l'addestramento d'accapo. Altrimenti riparte dall'ultimo checkpoint
        # overwrite_output_dir=True,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stop)]

    # trainer
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=valid_ds,
        tokenizer=tokenizer, callbacks=callbacks
    )
    # qui viene chiamata anche la funzione di loss
    trainer.train()
    print(trainer.evaluate())
    # è il checkpoint che ha ottenuto il valore migliore di eval_loss, 
    # non necessariamente l’ultimo.
    trainer.save_model("ckpt/dry_run/final_model") 

    # --------------------------------------------------------------------------------------------------------
    
    # Validazione del modello con validation_raw e no_match_validation
    name = "no_match_validation.tsv"
    filename = "data/dry_run/"+name
    df_final = pd.read_csv(filename, sep="\t")
    basename = os.path.basename(filename)
    print(f"Validazione finale su {basename}")
    final_ds = Dataset.from_pandas(df_final).map(tokenize_batch, batched=True)
    final_metrics = trainer.evaluate(eval_dataset=final_ds)
    print()
    print(f"Risultati validazione finale su {name}:", final_metrics)
    print()

    name = "validation_raw.tsv"
    filename = "data/dry_run/"+name
    df_final = pd.read_csv(filename, sep="\t")
    basename = os.path.basename(filename)
    print(f"Validazione finale su {basename}")
    final_ds = Dataset.from_pandas(df_final).map(tokenize_batch, batched=True)
    final_metrics = trainer.evaluate(eval_dataset=final_ds)
    print()
    print(f"Risultati validazione finale su {name}:", final_metrics)
    print()

if __name__=="__main__":
    main()


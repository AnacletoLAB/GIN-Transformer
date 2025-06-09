import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # comment this line for data parallel (train on all avaiables GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # comment this line for data parallel (train on all avaiables GPUs)
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from data import build_datasets, load_pairs_tsv
from tokenizer import tokenize_batch, tokenizer
from model import NucConfig, NucTransformer
from utils import set_seed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem, falling back to numerical ordering")


def main(
    train_file="data/dry_run/train.tsv", valid_file="data/dry_run/valid.tsv",
    output_dir="ckpt/dry_run", logging_dir="ckpt/dry_run/logs",
    learning_rate=3e-5, train_batch_size=32, eval_batch_size=32,
    num_epochs=10, bf16=True, grad_acc_steps=2, warmup_steps=1000,
    scheduler="cosine", early_stop=3, num_workers=8, seed=42
):
    set_seed(seed)
    # data
    train_ds, valid_ds = build_datasets(train_file, valid_file)
    train_ds = train_ds.map(tokenize_batch, batched=True)
    valid_ds = valid_ds.map(tokenize_batch, batched=True)

    # model & config
    config = NucConfig(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, max_len=100)
    model  = NucTransformer(config)

    # training args
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
        load_best_model_at_end=True,             # <-- load best on early stop
        metric_for_best_model="eval_loss",       # <-- which metric to monitor
        greater_is_better=False,                 # <-- lower loss is better
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stop)]

    # trainer
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=valid_ds,
        tokenizer=tokenizer, callbacks=callbacks
    )

    trainer.train()
    print(trainer.evaluate())

if __name__=="__main__":
    main()


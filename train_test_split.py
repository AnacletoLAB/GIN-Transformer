# da eseguire prima di train.py per suddividere il dataset in train e validation. Il dataset di validation viene ulteriormente filtrato
# con BLASTn. Viene ricavato un ulteriore dataset di validation filtrato no_match_validation.tsv per la validazione finale del modello
import os
import re
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split

# carico i dati del dataset da cui sono già state eliminate le sequenze di lunghezza maggiore di 127
# nucleotidi, dopodiché il dataset train_maxlen127.tsv viene suddiviso in train e validation
# il set di validazione df_val è il 10%
df = pd.read_csv("data/dry_run/train_maxlen127.tsv", sep="\t")
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# salvo su file .tsv il dataset che ho suddiviso in train e validation.
# il file di validation dovrà essere filtrato con l'algoritmo BLASTn
# l'output viene chiamato validation_raw.tsv perché il file dovrà essere splittato in due parti
# con reset_index() lascio inalterati gli indici originali
df_train.to_csv("data/dry_run/train_final.tsv", sep="\t", index= False)
df_val = df_val.reset_index()
df_val.to_csv("data/dry_run/validation_raw.tsv", sep="\t", index= False)
df_val.set_index("index", inplace=True)

# funzione per salvare il dataset in formato fasta
# e inoltre, le due sequenze vengono concatenate: seq1 + seq2
def to_fasta(df, filename):
    with open(f"{filename}.fasta", "w") as f:
        for idx, row in df.iterrows():
        # for idx, row in enumerate(df.itertuples(index=False)):
            sequenza_concatenata = row.seq1 + row.seq2
            f.write(f">{filename}_{idx}\n{sequenza_concatenata}\n")
            
# salvo i file in formato fasta
to_fasta(df_train, "data/dry_run/train_final")
to_fasta(df_val, "data/dry_run/validation_raw")
print("Creato il file train_final.tsv")
print("Creato il file validation_raw.tsv")

# --------------------------------------------------------------------------------------------

# Creazione del database BLAST, riga successiva: lancio BLASTn
# il file di output validation_blast.tsv contiene le coppie di validation filtrate tramite l'algoritmo BLASTn
os.system("makeblastdb -in data/dry_run/train_final.fasta -dbtype nucl -out blast_train_db")
os.system("blastn -query data/dry_run/validation_raw.fasta -db blast_train_db -out data/dry_run/validation_blast.tsv -outfmt '6 qseqid sseqid pident qcovs' -max_target_seqs 1")

# leggo il file appena creato e formatto la colonna eliminando il percorso del file
blast_df = pd.read_csv("data/dry_run/validation_blast.tsv", sep="\t", header=None, names=["qseqid", "sseqid", "pident", "qcovs"])
blast_df["qseqid"] = blast_df["qseqid"].apply(os.path.basename)
blast_df["sseqid"] = blast_df["sseqid"].apply(os.path.basename)

blast_df.to_csv("data/dry_run/validation_blast.tsv", sep="\t", index=False)
print(f"Creato il file validation_blast.tsv con {len(blast_df)} righe")
print()

# -------------------------------------------------------------------------------------------

# Carico in memoria i file validation_raw.tsv e validation_blast.tsv
df_raw = pd.read_csv("data/dry_run/validation_raw.tsv", sep="\t")
blast = pd.read_csv("data/dry_run/validation_blast.tsv", sep="\t")

# Estraggo gli indici numerici da qseqid (es. validation_raw_2574 → 2574)
blast["index"] = blast["qseqid"].str.extract(r"validation_raw_(\d+)$")[0].astype(int)
blast_indices = blast["index"].drop_duplicates().sort_values()

# salvo le sequenze corrispondenti agli indici BLAST in validation_final.tsv
df_blast = df_raw[df_raw["index"].isin(blast_indices)].reset_index(drop=True)
df_blast.to_csv("data/dry_run/validation_final.tsv", sep="\t", index=False)

# Creo no_match_validation.tsv 
# Sotto rimuovo le colonne inutili come level_0 o Unnamed
df_no_match = df_raw[~df_raw["index"].isin(blast_indices)].copy()
df_no_match = df_no_match.drop(columns=[col for col in df_no_match.columns if col in ["level_0", "Unnamed: 0"]])

# Prendo la colonna "index" e la uso come indice del dataframe
df_no_match.set_index("index", inplace=True)
df_no_match.to_csv("data/dry_run/no_match_validation.tsv", sep="\t", index=True)

print(f"File validation_final.tsv con {len(df_blast)} righe (è il validation_blast da cui sono state estratte le sequenze)")
print(f"File no_match_validation.tsv con {len(df_no_match)} righe (è il dataset filtrato ossia le sequenze non incluse in validation_blast.tsv)")
print(f"Totale righe in validation_raw.tsv: {len(df_raw)}")
print()

# Elimino la colonna index dai dataset (ora non mi serve più)
df = pd.read_csv("data/dry_run/validation_raw.tsv", sep="\t")
df = df[["seq1", "seq2"]]
df.to_csv("data/dry_run/validation_raw.tsv", sep="\t", index=False)

df = pd.read_csv("data/dry_run/no_match_validation.tsv", sep="\t")
df = df[["seq1", "seq2"]]
df.to_csv("data/dry_run/no_match_validation.tsv", sep="\t", index=False)

df = pd.read_csv("data/dry_run/validation_final.tsv", sep="\t")
df = df[["seq1", "seq2"]]
df.to_csv("data/dry_run/validation_final.tsv", sep="\t", index=False)











import pandas as pd
import os

# Pfade basierend auf deiner Struktur
base_path = "data/FSD50K/FSD50K.ground_truth"
vocab_path = os.path.join(base_path, "vocabulary.csv")
dev_csv_path = os.path.join(base_path, "dev.csv")
eval_csv_path = os.path.join(base_path, "eval.csv")

# 1. Vokabular laden, um die richtigen Begriffe zu finden
vocab = pd.read_csv(vocab_path, header=None, names=['index', 'label', 'm_code'])

# Wir suchen nach Labels, die "Glass" enthalten
glass_labels = vocab[vocab['label'].str.contains("Glass|Shatter|Crush", case=False)]
print("--- Gefundene relevante Klassen im Vokabular ---")
print(glass_labels)
print("\n")

# 2. Dev-Metadaten laden
df_dev = pd.read_csv(dev_csv_path)
df_eval = pd.read_csv(eval_csv_path)

# Liste der Kombinationen, die wir prüfen wollen
# Hinweis: Wir nutzen 'str.contains', da in FSD50K Labels oft kommagetrennt sind.
target_pairs = [
    ("Glass", "Shatter"),
    ("Glass", "Crushing"),
    ("Glass", "Crack"),
    ("Glass", "Slam"),
    ("Glass", "Crackle")
]

print(f"{'Kombination (dev)':<30} | {'Anzahl Clips':<12}")
print("-" * 45)

for main_label, sub_label in target_pairs:
    # Filter: Zeile muss beide Labels enthalten
    # Wir nutzen Wortgrenzen oder exakten Split, um Teilwort-Fehler zu vermeiden
    count_dev = df_dev[df_dev['labels'].apply(lambda x: main_label in x.split(',') and sub_label in x.split(','))].shape[0]
    
    print(f"{main_label} UND {sub_label:<20} | {count_dev:<12}")
    
# Gesamtzahl der Clips, die zumindest "Glass" enthalten
total_glass = df_dev[df_dev['labels'].apply(lambda x: "Glass" in x.split(','))].shape[0]
print("-" * 45)
print(f"{'Gesamtanzahl Glass (dev)':<30} | {total_glass:<12}")
print("\n")

# 3. Dasselbe für eval.csv
print(f"{'Kombination (eval)':<30} | {'Anzahl Clips':<12}")
print("-" * 45) 
for main_label, sub_label in target_pairs:
    count_eval = df_eval[df_eval['labels'].apply(lambda x: main_label in x.split(',') and sub_label in x.split(','))].shape[0]
    print(f"{main_label} UND {sub_label:<20} | {count_eval:<12}")

total_glass = df_eval[df_eval['labels'].apply(lambda x: "Glass" in x.split(','))].shape[0]
print("-" * 45)
print(f"{'Gesamtanzahl Glass (eval)':<30} | {total_glass:<12}")
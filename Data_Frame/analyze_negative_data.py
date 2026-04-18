import pandas as pd

# Pfade
dev_csv = "data/FSD50K/FSD50K.ground_truth/dev.csv"
eval_csv = "data/FSD50K/FSD50K.ground_truth/eval.csv"

# Beides laden und zusammenfügen für den Gesamtüberblick
df = pd.concat([pd.read_csv(dev_csv), pd.read_csv(eval_csv)])

# 1. Definiere die Gruppen
similar_sounds = [
    "Chink_and_clink", "Coin_(dropping)", "Cutlery_and_silverware", 
    "Dishes_and_pots_and_pans", "Keys_jangling", "Crack", 
    "Crackle", "Crumpling_and_crinkling", "Crushing", "Tearing"
]

ambient_sounds = [
    "Mechanical_fan", "Traffic_noise_and_roadway_noise", "Wind", 
    "Rain", "Chatter", "Crowd"
]

def analyze_negatives(target_list, group_name):
    print(f"\n--- Analyse: {group_name} ---")
    print(f"{'Label':<35} | {'Anzahl (ohne Glas)':<15}")
    print("-" * 55)
    
    for label in target_list:
        # Filter: Label vorhanden UND "Glass" oder "Shatter" NICHT vorhanden
        mask = df['labels'].apply(lambda x: label in x.split(',') and 
                                 not any(g in x.split(',') for g in ['Glass', 'Shatter']))
        count = df[mask].shape[0]
        print(f"{label:<35} | {count:<15}")

# Ausführung
analyze_negatives(similar_sounds, "ÄHNLICHE GERÄUSCHE (Hard Negatives)")
analyze_negatives(ambient_sounds, "UMGEBUNGSGERÄUSCHE (Background)")
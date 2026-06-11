# src/data_prep_positive.py
import os
import glob
import random
import shutil
from tqdm import tqdm

# --- 1. Kugelsichere Pfade erstellen ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

def main():
    # --- 2. Konfiguration für den Split ---
    TRAIN_RATIO = 0.8  # 80% Training, 20% Validierung (Test)
    
    # Wo liegen deine originalen, ungeschnittenen positiven Dateien?
    # Lege sie am besten alle in diesen Ordner:
    ordner_raw_pos = os.path.join(PROJECT_ROOT, "data", "raw", "positives")
    
    # Wo sollen sie hin?
    ordner_train_pos = os.path.join(PROJECT_ROOT, "data", "processed", "train", "pos")
    ordner_val_pos   = os.path.join(PROJECT_ROOT, "data", "processed", "val", "pos")
    
    # Erstelle die Ziel-Ordner, falls sie noch nicht existieren
    os.makedirs(ordner_train_pos, exist_ok=True)
    os.makedirs(ordner_val_pos, exist_ok=True)
    
    # --- 3. Dateien sammeln und mischen ---
    # Finde alle .wav Dateien im raw-Ordner
    alle_pos_dateien = glob.glob(os.path.join(ordner_raw_pos, "*.wav"))
    
    if len(alle_pos_dateien) == 0:
        print(f"FEHLER: Keine .wav Dateien in {ordner_raw_pos} gefunden!")
        print("Bitte lege deine originalen positiven 1,28s-Clips dorthin.")
        return
        
    print(f"Gefunden: {len(alle_pos_dateien)} positive Dateien.")
    
    # Zufällig mischen (extrem wichtig für einen echten Random-Split!)
    random.seed(42) # Fester Seed, damit der Split bei mehrmaligem Ausführen gleich bleibt
    random.shuffle(alle_pos_dateien)
    
    # --- 4. Den Split berechnen ---
    split_index = int(len(alle_pos_dateien) * TRAIN_RATIO)
    
    train_dateien = alle_pos_dateien[:split_index]
    val_dateien = alle_pos_dateien[split_index:]
    
    print(f"Teile auf: {len(train_dateien)} fürs Training ({(TRAIN_RATIO*100):.0f}%) | {len(val_dateien)} für Validierung ({((1-TRAIN_RATIO)*100):.0f}%)")
    
    # --- 5. Dateien kopieren ---
    print("\nKopiere Training-Dateien...")
    for datei in tqdm(train_dateien):
        dateiname = os.path.basename(datei)
        ziel_pfad = os.path.join(ordner_train_pos, dateiname)
        shutil.copy2(datei, ziel_pfad) # copy2 behält die Metadaten der Datei
        
    print("\nKopiere Validierungs-Dateien...")
    for datei in tqdm(val_dateien):
        dateiname = os.path.basename(datei)
        ziel_pfad = os.path.join(ordner_val_pos, dateiname)
        shutil.copy2(datei, ziel_pfad)
        
    print("\nKopieren erfolgreich abgeschlossen!")
    print("Dein Dataset ist nun komplett vorbereitet und bereit für das Training.")

if __name__ == "__main__":
    main()
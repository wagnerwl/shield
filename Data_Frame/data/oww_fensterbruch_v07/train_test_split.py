import os
import shutil
import random
import glob

# ==========================================
# 1. DEINE QUELL-ORDNER 
# (Hier liegen die fertig zugeschnittenen 1,28s Dateien)
# ==========================================
SOURCE_POS = "Data_Frame/data/DataSet_Glassbruch_all_v03/positive_samples"
SOURCE_NEG = "Data_Frame/data/DataSet_Glassbruch_all_v03/negative_samples"

# ==========================================
# 2. DEIN ZIEL-ORDNER (OpenWakeWord Struktur)
# ==========================================
# Das ist der Pfad, den das Skript aus deiner config generiert:
TARGET_BASE = "Data_Frame/data/oww_fensterbruch_v07"

# ==========================================
# 3. SPLIT-EINSTELLUNGEN
# ==========================================
TEST_SIZE = 0.15  # 15 % der Daten für den Test, 85 % für das Training

def split_and_copy(source_dir, target_train_dir, target_test_dir, label_name):
    # Alle .wav Dateien im Quellordner finden
    wav_files = glob.glob(os.path.join(source_dir, "*.wav"))
    
    if not wav_files:
        print(f"⚠️ Warnung: Keine .wav Dateien in '{source_dir}' gefunden!")
        return 0, 0
        
    # GANZ WICHTIG: Die Liste zufällig durchmischen!
    # Damit verhindern wir, dass z.B. alle klirrenden Flaschen im Training 
    # und alle dumpfen Flaschen im Test landen.
    random.seed(42) # Fester Seed, damit der Split reproduzierbar bleibt
    random.shuffle(wav_files)
    
    # Den Index berechnen, an dem wir die Liste durchschneiden
    split_index = int(len(wav_files) * (1 - TEST_SIZE))
    
    # Listen aufteilen
    train_files = wav_files[:split_index]
    test_files = wav_files[split_index:]
    
    # Ordner erstellen, falls sie nicht existieren (löscht keine alten Dateien)
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_test_dir, exist_ok=True)
    
    # Dateien ins Training kopieren
    for f in train_files:
        shutil.copy2(f, os.path.join(target_train_dir, os.path.basename(f)))
        
    # Dateien in den Test kopieren
    for f in test_files:
        shutil.copy2(f, os.path.join(target_test_dir, os.path.basename(f)))
        
    print(f"✅ {label_name} erfolgreich aufgeteilt:")
    print(f"   -> {len(train_files)} Dateien in {target_train_dir}")
    print(f"   -> {len(test_files)} Dateien in {target_test_dir}")
    
    return len(train_files), len(test_files)

def main():
    print("Starte Train-Test-Split...\n")
    
    # Zielordner definieren
    pos_train_dir = os.path.join(TARGET_BASE, "positive_train")
    pos_test_dir = os.path.join(TARGET_BASE, "positive_test")
    neg_train_dir = os.path.join(TARGET_BASE, "negative_train")
    neg_test_dir = os.path.join(TARGET_BASE, "negative_test")
    
    # 1. Positive Daten (Fensterbrüche) splitten
    pos_train_count, pos_test_count = split_and_copy(
        SOURCE_POS, pos_train_dir, pos_test_dir, "Fensterbrüche (Positiv)"
    )
    
    print("-" * 40)
    
    # 2. Negative Daten (Flaschen/Störgeräusche) splitten
    neg_train_count, neg_test_count = split_and_copy(
        SOURCE_NEG, neg_train_dir, neg_test_dir, "Störgeräusche (Negativ)"
    )
    
    print("\nZusammenfassung für deine fenster_config.yml:")
    print("-------------------------------------------")
    print(f"n_samples: {pos_train_count}")
    print(f"n_samples_val: {pos_test_count}")
    print("\nDie Daten liegen jetzt perfekt vorbereitet in deiner OpenWakeWord-Ordnerstruktur!")

if __name__ == "__main__":
    main()
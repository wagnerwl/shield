import os
import shutil
import random

# Pfade definieren
BASE_DIR = "./data/embeddings"
CATEGORIES = ["positive", "negative"]
SPLIT_RATIO = 0.8  # 80% Training, 20% Validierung

# Zufall festlegen, damit das Ergebnis reproduzierbar ist
random.seed(42)

def setup_folders():
    for split in ["train", "val"]:
        for cat in CATEGORIES:
            path = os.path.join(BASE_DIR, split, cat)
            os.makedirs(path, exist_ok=True)

def move_files():
    setup_folders()
    
    for cat in CATEGORIES:
        source_path = os.path.join(BASE_DIR, cat)
        
        # Falls die Dateien schon in Unterordnern sind, nichts tun
        if not os.path.exists(source_path):
            print(f"Hinweis: Quellordner {source_path} existiert nicht (vielleicht schon verschoben?)")
            continue

        files = [f for f in os.listdir(source_path) if f.endswith('.npy')]
        random.shuffle(files)

        split_idx = int(len(files) * SPLIT_RATIO)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(f"Kategorie {cat}: Verschiebe {len(train_files)} ins Training und {len(val_files)} in die Validierung...")

        for f in train_files:
            shutil.move(os.path.join(source_path, f), os.path.join(BASE_DIR, "train", cat, f))
        
        for f in val_files:
            shutil.move(os.path.join(source_path, f), os.path.join(BASE_DIR, "val", cat, f))

        # Alten leeren Ordner entfernen
        try:
            os.rmdir(source_path)
        except:
            pass

if __name__ == "__main__":
    move_files()
    print("\nDaten erfolgreich physisch getrennt!")
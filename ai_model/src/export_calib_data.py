# src/export_calib_data.py
import os
import numpy as np
import torch
from dataset import SoundDataset

# Pfade sauber aufbauen
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Wir nutzen einfach den Validierungs-Datensatz, da ist keine Augmentation (Verzerrung) drauf
val_dir = os.path.join(PROJECT_ROOT, "data", "processed", "val")
export_dir = os.path.join(PROJECT_ROOT, "data", "calibration_samples")

# Ordner erstellen, falls er nicht existiert
os.makedirs(export_dir, exist_ok=True)

print("Lade Validierungs-Datensatz...")
# is_train=False ist wichtig, damit kein künstlicher Raumhall (RIR) dazukommt!
dataset = SoundDataset(val_dir, is_train=False)

# Wir brauchen ca. 100 Spektrogramme für TensorFlow
max_samples = min(100, len(dataset))

print(f"Exportiere {max_samples} echte Spektrogramme nach '{export_dir}'...")

for i in range(max_samples):
    mel_spec, label = dataset[i]
    
    # mel_spec ist ein PyTorch Tensor der Form [1, 64, 41] (Channels, Mels, Frames)
    mel_np = mel_spec.numpy()
    
    # WICHTIG: PyTorch nutzt [Channels, Höhe, Breite]. 
    # TensorFlow nutzt [Höhe, Breite, Channels]. 
    # Wir müssen das Array drehen (transpose), damit TensorFlow es später versteht.
    # Aus [1, 64, 41] wird [64, 41, 1]
    mel_tf = np.transpose(mel_np, (1, 2, 0))
    
    # TensorFlow erwartet immer einen "Batch"-Rahmen ganz außen, also [1, 64, 41, 1]
    mel_tf = np.expand_dims(mel_tf, axis=0)
    
    # Als .npy Datei speichern
    speicher_pfad = os.path.join(export_dir, f"sample_{i:03d}.npy")
    np.save(speicher_pfad, mel_tf)

print("✅ Fertig! Die Kalibrierungs-Daten liegen bereit.")
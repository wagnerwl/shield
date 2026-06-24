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
    mel_np = mel_spec.numpy()
    
    # WICHTIG: Kein np.transpose mehr! Wir bleiben strikt im PyTorch-Format.
    # Aus [1, 64, 41] wird direkt [1, 1, 64, 41]
    mel_tf = np.expand_dims(mel_np, axis=0)
    
    speicher_pfad = os.path.join(export_dir, f"sample_{i:03d}.npy")
    np.save(speicher_pfad, mel_tf)

print("✅ Fertig! Die Kalibrierungs-Daten liegen bereit.")
# src/dataset.py
import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import yaml

# Kugelsichere Pfade
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

class SoundDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: Pfad zum Ordner (z.B. 'data/processed/train')
        Erwartet Unterordner: 'pos', 'neg_bg', 'neg_hard'
        """
        self.file_paths = []
        self.labels = []
        self.sample_weights = [] # Hier speichern wir das Gewicht für den Sampler
        
        pos_dir = os.path.join(data_dir, "pos")
        neg_bg_dir = os.path.join(data_dir, "neg_bg")
        neg_hard_dir = os.path.join(data_dir, "neg_hard")
        
        # Alle .wav Dateien finden
        pos_files = glob.glob(os.path.join(pos_dir, "*.wav"))
        neg_bg_files = glob.glob(os.path.join(neg_bg_dir, "*.wav"))
        neg_hard_files = glob.glob(os.path.join(neg_hard_dir, "*.wav"))
        
        n_pos = len(pos_files)
        n_bg = len(neg_bg_files)
        n_hard = len(neg_hard_files)
        
        print(f"Gefunden: {n_pos} Positiv | {n_bg} bg Negativ | {n_hard} Hard Negativ")
        
        # --- DER TRICK: Die Gewichte berechnen ---
        # Ziel-Verteilung im Batch: 50% Positiv, 25% bg Negativ, 25% Hard Negativ
        weight_pos = 0.5 / n_pos if n_pos > 0 else 0
        weight_bg = 0.25 / n_bg if n_bg > 0 else 0
        weight_hard = 0.25 / n_hard if n_hard > 0 else 0
        
        # Daten und Gewichte in die Listen eintragen
        for f in pos_files:
            self.file_paths.append(f)
            self.labels.append(1.0)
            self.sample_weights.append(weight_pos)
            
        for f in neg_bg_files:
            self.file_paths.append(f)
            self.labels.append(0.0) # Label ist trotzdem 0 (Negativ)
            self.sample_weights.append(weight_bg)
            
        for f in neg_hard_files:
            self.file_paths.append(f)
            self.labels.append(0.0) # Label ist trotzdem 0 (Negativ)
            self.sample_weights.append(weight_hard)
        
        # Spektrogramm-Wandler
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['audio']['sample_rate'],
            n_mels=config['audio']['n_mels'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length']
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # ==========================================
        # 1. Volume Augmentation (Nur im Training!)
        # ==========================================
        # Mit 80% Wahrscheinlichkeit verändern wir die Lautstärke zufällig
        import random
        if random.random() < 0.8:
            # Wählt einen zufälligen Faktor zwischen 0.1 (sehr leise) und 1.2 (etwas lauter)
            gain = random.uniform(0.1, 1.2)
            waveform = waveform * gain
            
        # ==========================================
        # 2. Spektrogramm erstellen
        # ==========================================
        mel_spec = self.mel_transform(waveform)
        
        # ==========================================
        # 3. Normalisierung (Der wichtigste Schritt gegen Klatschen!)
        # ==========================================
        # Zieht den Mittelwert ab und teilt durch die Standardabweichung.
        # Ein Klatschen und ein Glasbruch haben nun dieselbe "Grundhelligkeit", 
        # das Netz MUSS jetzt auf die spezifischen Splitter/Klirr-Frequenzen achten.
        mel_spec_mean = mel_spec.mean()
        mel_spec_std = mel_spec.std()
        
        # Das + 1e-6 verhindert, dass wir durch Null teilen, falls absolute Stille herrscht
        mel_spec = (mel_spec - mel_spec_mean) / (mel_spec_std + 1e-6) 
        
        # Label bereithalten
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return mel_spec, label
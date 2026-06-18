# src/dataset.py
import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import yaml
import random # Am besten gleich oben importieren

# Kugelsichere Pfade
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

class SoundDataset(Dataset):
    # ==========================================
    # ÄNDERUNG 1: is_train Flag hinzugefügt
    # ==========================================
    def __init__(self, data_dir, is_train=True):
        """
        data_dir: Pfad zum Ordner (z.B. 'data/processed/train')
        is_train: True für Training (mit Augmentation), False für Validierung/Test
        """
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.sample_weights = [] 
        
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
        
        print(f"Gefunden: {n_pos} Positiv | {n_bg} bg Negativ | {n_hard} Hard Negativ (is_train={is_train})")
        
        # Gewichte berechnen
        weight_pos = 0.3 / n_pos if n_pos > 0 else 0
        weight_bg = 0.35 / n_bg if n_bg > 0 else 0
        weight_hard = 0.35 / n_hard if n_hard > 0 else 0
        
        for f in pos_files:
            self.file_paths.append(f)
            self.labels.append(1.0)
            self.sample_weights.append(weight_pos)
            
        for f in neg_bg_files:
            self.file_paths.append(f)
            self.labels.append(0.0)
            self.sample_weights.append(weight_bg)
            
        for f in neg_hard_files:
            self.file_paths.append(f)
            self.labels.append(0.0) 
            self.sample_weights.append(weight_hard)
        
        # Spektrogramm-Wandler
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['audio']['sample_rate'],
            n_mels=config['audio']['n_mels'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length']
        )
        
        # ==========================================
        # ÄNDERUNG 2: Maskierungs-Tools vorbereiten
        # ==========================================
        # Wir definieren hier, wie groß die schwarzen Balken maximal sein dürfen
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # ==========================================
        # ÄNDERUNG 3: Volume Augmentation schützen
        # ==========================================
        if self.is_train:
            if random.random() < 0.8:
                gain = random.uniform(0.1, 1.2)
                waveform = waveform * gain
            
        # Spektrogramm erstellen
        mel_spec = self.mel_transform(waveform)
        
        # Normalisierung
        mel_spec_mean = mel_spec.mean()
        mel_spec_std = mel_spec.std()
        mel_spec = (mel_spec - mel_spec_mean) / (mel_spec_std + 1e-6) 
        
        # ==========================================
        # ÄNDERUNG 4: SpecAugment anwenden
        # ==========================================
        # Zieht zufällige horizontale (Frequenz) und vertikale (Zeit) Balken über das Bild.
        # So lernt das Modell, das Geräusch auch dann zu erkennen, wenn Teile davon
        # überlagert werden oder kurz fehlen.
        if self.is_train:
            if random.random() < 0.5:
                mel_spec = self.freq_masking(mel_spec)
            if random.random() < 0.5:
                mel_spec = self.time_masking(mel_spec)
        
        # Label bereithalten
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return mel_spec, label
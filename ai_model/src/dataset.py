# src/dataset.py
import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio # NEU FÜR RIR
from torch.utils.data import Dataset
import yaml
import random

# Kugelsichere Pfade
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR) # Entspricht dem ai_model Ordner
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

class SoundDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        """
        data_dir: Pfad zum Ordner (z.B. 'data/processed/train')
        is_train: True für Training (mit Augmentation), False für Validierung/Test
        """
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self.sample_weights = [] 
        
        # ==========================================
        # NEU: RIR Dateien laden (nur für Training)
        # ==========================================
        self.rir_files = []
        if self.is_train:
            # Pfad zum RIR-Datensatz: ai_model/data/rir_dataset
            rir_dir = os.path.join(PROJECT_ROOT, "data", "rir_dataset", "convert") 
            self.rir_files = glob.glob(os.path.join(rir_dir, "*.wav"))
            print(f"RIR Dateien gefunden: {len(self.rir_files)}")

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
        self.db_transform = T.AmplitudeToDB() # NEU HINZUFÜGEN!
        
        # Maskierungs-Tools vorbereiten
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        original_len = waveform.shape[1] # Länge merken für RIR!
        
        # ==========================================
        # NEU: RIR Augmentation anwenden
        # ==========================================
        if self.is_train and len(self.rir_files) > 0:
            if random.random() < 0.4: # In 40% der Fälle einen Raumhall anwenden
                rir_pfad = random.choice(self.rir_files)
                rir_waveform, rir_sr = torchaudio.load(rir_pfad)
                
                # Resampling, falls RIR eine andere Sample-Rate hat
                if rir_sr != sr:
                    rir_waveform = T.Resample(rir_sr, sr)(rir_waveform)
                
                # Auf Mono mischen, falls Stereo
                if rir_waveform.shape[0] > 1:
                    rir_waveform = rir_waveform.mean(dim=0, keepdim=True)
                    
                # RIR normalisieren (verhindert extreme Lautstärkesprünge)
                norm = torch.norm(rir_waveform, p=2)
                if norm > 0:
                    rir_waveform = rir_waveform / norm
                
                # Faltung (Convolution) anwenden
                waveform = F_audio.fftconvolve(waveform, rir_waveform)
                
                # WICHTIG: Durch die Faltung wird die Waveform länger. 
                # Wir müssen sie wieder auf die Original-Länge abschneiden!
                waveform = waveform[:, :original_len]

        # Volume Augmentation schützen
        if self.is_train:
            if random.random() < 0.8:
                gain = random.uniform(0.1, 1.2)
                waveform = waveform * gain
            
        # Spektrogramm erstellen
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec) # NEU: Komprimiert die Dynamik!

        # Normalisierung
        mel_spec_mean = mel_spec.mean()
        mel_spec_std = mel_spec.std()
        mel_spec = (mel_spec - mel_spec_mean) / (mel_spec_std + 1e-6) 
        
        # SpecAugment anwenden
        if self.is_train:
            if random.random() < 0.5:
                mel_spec = self.freq_masking(mel_spec)
            if random.random() < 0.5:
                mel_spec = self.time_masking(mel_spec)
        
        # Label bereithalten
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return mel_spec, label
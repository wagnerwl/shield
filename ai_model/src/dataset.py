# src/dataset.py
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import yaml

# Lade die Config-Datei
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

class SoundDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        
        # Greife auf die Werte aus der YAML-Datei zu
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config['audio']['sample_rate'],
            n_mels=config['audio']['n_mels'],
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length']
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Lade das Audio
        waveform, sr = torchaudio.load(self.file_paths[idx])
        
        # 2. Wandle in Spektrogramm um
        mel_spec = self.mel_transform(waveform)
        
        # 3. Label bereithalten
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return mel_spec, label
# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from model import SoundDetectorCNN
from dataset import SoundDataset

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss berechnen
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # pt ist die Wahrscheinlichkeit für die richtige Klasse
        pt = torch.exp(-bce_loss)
        # Focal Loss Formel anwenden
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Kugelsichere Pfade
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

def main():
    # ==========================================
    # NEU: Apple Silicon (M1/M2/M3) GPU aktivieren
    # ==========================================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Nutze Apple M2 GPU (MPS) für das Training!")
    else:
        device = torch.device("cpu")
        print("Nutze CPU...")

    train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "train")
    
    print("Lade Dataset und berechne Gewichte...")
    dataset = SoundDataset(train_dir)
    
    EPOCH_SIZE = 30000 
    
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=EPOCH_SIZE,
        replacement=True 
    )
    
    # ==========================================
    # NEU: num_workers und pin_memory hinzugefügt
    # ==========================================
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler,
        num_workers=2,      # 2 Kerne laden die Daten im Hintergrund
        pin_memory=True     # Schiebt Daten schneller zur GPU
    )
   
    val_dir = os.path.join(PROJECT_ROOT, "data", "processed", "val")
    val_dataset = SoundDataset(val_dir, is_train=False)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=2,      # 2 Kerne laden die Daten im Hintergrund
        pin_memory=True     # Schiebt Daten schneller zur GPU
    )

    # Modell initialisieren UND auf das Gerät (GPU) schieben
    modell = SoundDetectorCNN().to(device)
    
    # criterion = nn.BCELoss() 
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(modell.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    # ==========================================
    # NEU: Learning Rate Scheduler
    # ==========================================
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # Wir wollen den Zielwert (False Positives) minimieren
        factor=0.5,      # Wenn es nicht weitergeht: Lernrate halbieren
        patience=2       # Nach 2 Epochen ohne Verbesserung greift der Scheduler
    )
    
    best_fp_rate = float('inf')

    print("\nStarte Training...")
    for epoch in range(config['training']['epochs']):
        
        # ==========================
        # 1. TRAINING
        # ==========================
        modell.train() 
        running_train_loss = 0.0
        
        for mel_specs_batch, labels_batch in dataloader: 
            # NEU: Daten auf die GPU schieben
            mel_specs_batch = mel_specs_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()                 
            vorhersagen = modell(mel_specs_batch) 
            loss = criterion(vorhersagen, labels_batch) 
            loss.backward()                       
            optimizer.step()                      
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(dataloader)

        # ==========================
        # 2. VALIDIERUNG (METRIKEN)
        # ==========================
        modell.eval() 
        
        true_positives = 0
        false_positives = 0
        total_positives = 0
        total_negatives = 0
        
        with torch.no_grad(): 
            for val_mel, val_labels in val_dataloader:
                # NEU: Daten auf die GPU schieben
                val_mel = val_mel.to(device)
                val_labels = val_labels.to(device)
                
                vorhersagen = modell(val_mel)
                
                preds = (vorhersagen >= 0.8).float()
                
                true_positives += ((preds == 1) & (val_labels == 1)).sum().item()
                false_positives += ((preds == 1) & (val_labels == 0)).sum().item()
                
                total_positives += (val_labels == 1).sum().item()
                total_negatives += (val_labels == 0).sum().item()

        # --- Metriken berechnen ---
        richtig_erkannt = true_positives + (total_negatives - false_positives)
        val_accuracy = richtig_erkannt / (total_positives + total_negatives) if (total_positives + total_negatives) > 0 else 0
        
        val_recall = true_positives / total_positives if total_positives > 0 else 0
        
        stunden_negativ = (total_negatives * config['audio']['clip_length_seconds']) / 3600
        fp_per_hour = false_positives / stunden_negativ if stunden_negativ > 0 else 0

        print(f"\n--- Epoch {epoch+1:02d}/{config['training']['epochs']} ---")
        print(f"Training Loss:         {avg_train_loss:.4f}")
        print(f"Validation Recall:     {val_recall*100:.1f} %")
        print(f"False Positives/Hour:  {fp_per_hour:.2f}")
        print(f"Learning Rate:         {optimizer.param_groups[0]['lr']:.6f}")

        # ==========================================
        # NEU: Dem Scheduler den aktuellen Wert übergeben
        # ==========================================
        scheduler.step(fp_per_hour)

        if fp_per_hour < best_fp_rate and val_recall > 0.85:
            best_fp_rate = fp_per_hour
            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            speicher_pfad = os.path.join(models_dir, "mein_geraeusch_cnn_best.pt")
            torch.save(modell.state_dict(), speicher_pfad)
            print("🌟 Neues bestes Modell gespeichert!")

    # Modell am Ende speichern
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    speicher_pfad = os.path.join(models_dir, "mein_geraeusch_cnn.pt")
    print(f"\nTraining beendet! Modell gespeichert unter: {speicher_pfad}")

if __name__ == "__main__":
    main()
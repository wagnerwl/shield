# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from model import SoundDetectorCNN
from dataset import SoundDataset

# Kugelsichere Pfade
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

def main():
    train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "train")
    
    print("Lade Dataset und berechne Gewichte...")
    dataset = SoundDataset(train_dir)
    
    # --- DER SAMPLER ---
    # Da wir nun mit Wahrscheinlichkeiten ziehen, müssen wir sagen, 
    # wie viele Clips wir pro Epoche insgesamt sehen wollen.
    # Z.B. 10.000 Ziehungen bilden eine Epoche.
    EPOCH_SIZE = 10000 
    
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=EPOCH_SIZE,
        replacement=True # GANZ WICHTIG! Erlaubt es, ein positives Sample mehrfach pro Epoche zu ziehen
    )
    
    # Der DataLoader bekommt jetzt den Sampler übergeben.
    # WICHTIG: shuffle=False! Der Sampler übernimmt ab sofort das Mischen.
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler
    )
   
    val_dir = os.path.join(PROJECT_ROOT, "data", "processed", "val")
    val_dataset = SoundDataset(val_dir)
    # Beim Validieren brauchen wir keinen Sampler und kein Shuffle, 
    # wir testen einfach stur alle Dateien einmal durch.
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

     
    modell = SoundDetectorCNN()
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(modell.parameters(), lr=config['training']['learning_rate'])

    best_fp_rate = float('inf') # Startet bei unendlich

    print("\nStarte Training...")
    for epoch in range(config['training']['epochs']):
        
        # ==========================
        # 1. TRAINING
        # ==========================
        modell.train() # Wichtig: Sagt dem Modell, dass es jetzt lernt
        running_train_loss = 0.0
        
        for mel_specs_batch, labels_batch in dataloader: # Hier läuft der Trainings-Sampler
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
        modell.eval() # Wichtig: Sagt dem Modell, dass es jetzt NICHT lernt, sondern nur testet
        
        true_positives = 0
        false_positives = 0
        total_positives = 0
        total_negatives = 0
        
        with torch.no_grad(): # Schaltet die Gradienten-Berechnung ab (spart Speicher & Zeit)
            for val_mel, val_labels in val_dataloader:
                vorhersagen = modell(val_mel)
                
                # Alles über 0.5 werten wir als "Geräusch erkannt" (1), darunter als "Hintergrund" (0)
                preds = (vorhersagen >= 0.5).float()
                
                # Richtig positiv (Modell sagt 1, Wahrheit ist 1)
                true_positives += ((preds == 1) & (val_labels == 1)).sum().item()
                # Falsch positiv / Fehlalarm (Modell sagt 1, Wahrheit ist 0)
                false_positives += ((preds == 1) & (val_labels == 0)).sum().item()
                
                total_positives += (val_labels == 1).sum().item()
                total_negatives += (val_labels == 0).sum().item()

        # --- Metriken berechnen ---
        # Accuracy: Wie viel Prozent aller Dateien (Pos + Neg) wurden richtig erkannt?
        richtig_erkannt = true_positives + (total_negatives - false_positives)
        val_accuracy = richtig_erkannt / (total_positives + total_negatives) if (total_positives + total_negatives) > 0 else 0
        
        # Recall: Von allen ECHTEN Geräuschen, wie viele hat das Modell gefunden?
        val_recall = true_positives / total_positives if total_positives > 0 else 0
        
        # False Positives pro Stunde: 
        # Wie oft schlägt das Modell bei Hintergrundgeräuschen fälschlicherweise Alarm?
        # Dauer aller negativen Validierungs-Clips berechnen:
        stunden_negativ = (total_negatives * config['audio']['clip_length_seconds']) / 3600
        fp_per_hour = false_positives / stunden_negativ if stunden_negativ > 0 else 0

        print(f"\n--- Epoch {epoch+1:02d}/{config['training']['epochs']} ---")
        print(f"Training Loss:         {avg_train_loss:.4f}")
        print(f"Validation Recall:     {val_recall*100:.1f} %")
        print(f"False Positives/Hour:  {fp_per_hour:.2f}")

        # Speichere das Modell NUR, wenn die Fehlalarme besser geworden sind 
        # (und der Recall trotzdem über z.B. 85% bleibt)
        if fp_per_hour < best_fp_rate and val_recall > 0.85:
            best_fp_rate = fp_per_hour
            models_dir = os.path.join(PROJECT_ROOT, "models")
            os.makedirs(models_dir, exist_ok=True)
            speicher_pfad = os.path.join(models_dir, "mein_geraeusch_cnn_best.pt")
            torch.save(modell.state_dict(), speicher_pfad)
            print("🌟 Neues bestes Modell gespeichert!")


    # Modell speichern
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    speicher_pfad = os.path.join(models_dir, "mein_geraeusch_cnn.pt")
    print(f"\nTraining beendet! Modell gespeichert unter: {speicher_pfad}")

if __name__ == "__main__":
    main()
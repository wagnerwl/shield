# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml
from model import SoundDetectorCNN
from dataset import SoundDataset

def main():
    # --- 1. Daten laden (Hier trägst du später deine echten Pfade ein) ---
    # Beispiel-Pfade, sobald deine data_prep.py die Audios zerschnitten hat
    file_paths = ["../data/processed/train/pos/clip1.wav", "../data/processed/train/neg/clip2.wav"]
    labels = [1, 0] # 1 für dein Geräusch, 0 für Hintergrund

    # --- 2. Dataset und DataLoader vorbereiten ---
    dataset = SoundDataset(file_paths, labels)
    # Der DataLoader mischt die Daten und baut Pakete (Batches)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # --- 3. Modell und Werkzeuge initialisieren ---
    modell = SoundDetectorCNN()
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(modell.parameters(), lr=config.LEARNING_RATE)

    # --- 4. Der Trainings-Loop ---
    print("Starte Training...")
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        
        for mel_specs_batch, labels_batch in dataloader:
            optimizer.zero_grad()                 
            vorhersagen = modell(mel_specs_batch) 
            loss = criterion(vorhersagen, labels_batch) 
            loss.backward()                       
            optimizer.step()                      
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.EPOCHS} beendet. Durchschnittlicher Loss: {running_loss/len(dataloader):.4f}")

    # --- 5. Modell speichern ---
    torch.save(modell.state_dict(), "../models/mein_sound_modell.pt")
    print("Modell erfolgreich gespeichert!")

if __name__ == "__main__":
    main()
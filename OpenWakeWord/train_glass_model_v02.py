import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Konfiguration
BASE_EMBEDDING_DIR = "./Data_Frame/data/embeddings"     #Hier liegen die npy Dateien
MODEL_NAME = "glass_break_model_v2"
BATCH_SIZE = 16 # Kleinere Batches helfen oft beim Fine-Tuning
EPOCHS = 150    # Höher, aber wir nutzen Early Stopping
LEARNING_RATE = 0.001
MODEL_DIR = "OpenWakeWord/Model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_temp.pth")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")

def load_from_directory(directory):
    X, y = [], []
    for label_name, label_value in [("positive", 1), ("negative", 0)]:
        folder = os.path.join(directory, label_name)
        if not os.path.exists(folder): continue
        
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        for f in files:
            data = np.load(os.path.join(folder, f)).squeeze()
            
            # VERBESSERUNG: Wir kombinieren Durchschnitt UND Maximum
            # Das hilft, kurze impulsive Geräusche (Glas) besser zu erkennen
            avg_feat = np.mean(data, axis=0)
            max_feat = np.max(data, axis=0)
            combined_feat = np.concatenate([avg_feat, max_feat]) # Jetzt 192 Features
            
            X.append(combined_feat)
            y.append(label_value)
    return np.array(X), np.array(y)

# Daten laden
X_train, y_train = load_from_directory(os.path.join(BASE_EMBEDDING_DIR, "train"))
X_val, y_val = load_from_directory(os.path.join(BASE_EMBEDDING_DIR, "val"))

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 2. Modell-Architektur (etwas komplexer für die kombinierten Features)
class ImprovedGlassClassifier(nn.Module):
    def __init__(self):
        super(ImprovedGlassClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(192, 128), # 192 wegen Mean + Max
            nn.ReLU(),
            nn.BatchNorm1d(128), # Stabilisiert das Training
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

device = torch.device("cpu") # Auf dem Mac meist am stabilsten
model = ImprovedGlassClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lernraten-Anpassung: Reduziert LR, wenn es nicht mehr weitergeht
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)

# 3. Training mit Early Stopping
best_acc = 0
patience_counter = 0
print(f"Starte verbessertes Training...")

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Validierung
    model.eval()
    val_preds = []
    with torch.no_grad():
        for b_val_X, _ in val_loader:
            outputs = model(b_val_X).squeeze()
            val_preds.extend((outputs > 0.5).float().numpy())
    
    acc = (np.array(val_preds) == y_val).mean()
    scheduler.step(acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoche [{epoch+1}/{EPOCHS}] - Val Acc: {acc:.4f} (Beste: {best_acc:.4f})")
    
    # Speichere das beste Modell
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Abbrechen, wenn 25 Epochen keine Verbesserung kam
    if patience_counter > 25:
        print(f"Early Stopping bei Epoche {epoch+1}")
        break

# 4. Export des BESTEN Standes
print(f"\nTraining beendet. Beste Genauigkeit: {best_acc:.4f}")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
dummy_input = torch.randn(1, 192)
torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"Verbessertes Modell als '{ONNX_MODEL_PATH}' gespeichert.")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Konfiguration
BASE_EMBEDDING_DIR = "./data/embeddings"
MODEL_NAME = "glass_break_model"
BATCH_SIZE = 32
EPOCHS = 60  # Etwas erhöht für bessere Konvergenz
LEARNING_RATE = 0.001

def load_from_directory(directory):
    """Lädt alle .npy Dateien aus einem Verzeichnis (positive & negative)"""
    X, y = [], []
    
    # Pfade für diese Gruppe (z.B. 'train' oder 'val')
    for label_name, label_value in [("positive", 1), ("negative", 0)]:
        folder = os.path.join(directory, label_name)
        if not os.path.exists(folder):
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        for f in files:
            data = np.load(os.path.join(folder, f))
            # Durchschnitt über die Zeitachse bilden (Pooling)
            # data.squeeze() macht aus (1, Frames, 96) -> (Frames, 96)
            avg_features = np.mean(data.squeeze(), axis=0)
            X.append(avg_features)
            y.append(label_value)
            
    return np.array(X), np.array(y)

# 2. Daten physisch laden
print("Lade Trainingsdaten...")
X_train, y_train = load_from_directory(os.path.join(BASE_EMBEDDING_DIR, "train"))

print("Lade Validierungsdaten...")
X_val, y_val = load_from_directory(os.path.join(BASE_EMBEDDING_DIR, "val"))

print(f"\nDatensatz-Statistik:")
print(f"  Training: {len(X_train)} Samples")
print(f"  Validierung: {len(X_val)} Samples")

# In PyTorch Tensoren umwandeln
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 3. Modell-Architektur
class GlassClassifier(nn.Module):
    def __init__(self):
        super(GlassClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Etwas mehr Schutz gegen Overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GlassClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Trainings-Schleife
print(f"\nStarte Training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validierung am Ende jeder Epoche
    model.eval()
    val_preds = []
    with torch.no_grad():
        for b_val_X, b_val_y in val_loader:
            b_val_X = b_val_X.to(device)
            outputs = model(b_val_X).squeeze()
            preds = (outputs > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
    
    acc = (np.array(val_preds) == y_val).mean()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoche [{epoch+1}/{EPOCHS}] - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")

# 5. Finale Evaluierung & Export
print("\nTraining abgeschlossen.")
print("Exportiere Modell nach ONNX...")

model.eval()
dummy_input = torch.randn(1, 96).to(device)
torch.onnx.export(model, dummy_input, f"{MODEL_NAME}.onnx", 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"Erfolg! Die Datei '{MODEL_NAME}.onnx' ist bereit für openWakeWord.")
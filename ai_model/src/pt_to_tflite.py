import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. DEINE AKTUELLE MODELLKLASSE DEFINIEREN
# ==========================================
class SoundDetectorCNN(nn.Module):
    def __init__(self):
        super(SoundDetectorCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # NEU: BatchNorm muss hier rein!
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # NEU: BatchNorm muss hier rein!
        self.pool2 = nn.MaxPool2d(2)
        
        # Fully Connected
        self.fc1 = nn.Linear(5120, 64) 
        self.dropout = nn.Dropout(p=0.6) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Reihenfolge: Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = torch.sigmoid(self.fc2(x)) 
        return x

# ==========================================
# SCHRITT 1: PyTorch zu ONNX
# ==========================================
print("Starte Schritt 1: PyTorch zu ONNX...")

model = SoundDetectorCNN()

# WICHTIG: Nutze hier am besten das gespeicherte Best-Modell!
# Passe den Pfad ggf. an, z.B. "ai_model/models/mein_geraeusch_cnn_best.pt"
modell_pfad = "ai_model/models/mein_geraeusch_cnn_014.pt" 

if not os.path.exists(modell_pfad):
    print(f"FEHLER: Konnte die Modelldatei nicht finden: {modell_pfad}")
    exit()

# Lädt die Gewichte
model.load_state_dict(torch.load(modell_pfad, map_location=torch.device('cpu')))
model.eval() 

# Der Dummy-Input: Shape [Batch, Channel, Mels, Zeit-Frames]
# 16000 Hz * 1.28 s = 20480 Samples. Bei Hop-Length 512 ergibt das exakt 41 Frames.
dummy_input = torch.randn(1, 1, 64, 41) 

onnx_ordner = "Arduino/onnx"
os.makedirs(onnx_ordner, exist_ok=True) # Ordner sicher erstellen
onnx_path = os.path.join(onnx_ordner, "audio_cnn.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print(f"-> Erfolgreich gespeichert als {onnx_path}")

# ==========================================
# SCHRITT 2: ONNX direkt zu TFLite (mit onnx2tf)
# ==========================================
print("\nStarte Schritt 2: ONNX zu TFLite...")

ausgabe_ordner = "Arduino/tflite_ausgabe"

# Falls der Ordner nicht existiert, wird er von onnx2tf angelegt
subprocess.run([
    "onnx2tf", 
    "-i", onnx_path, 
    "-o", ausgabe_ordner,
    "--keep_ncw_or_nchw_or_ncdhw_input_names", "input", # <--- DIESE ZEILE NEU HINZUFÜGEN!
    "--non_verbose" 
], check=True)

print(f"\n-> BÄM! Erfolgreich abgeschlossen!")
print(f"-> Dein fertiges TFLite Modell liegt im Ordner: '{ausgabe_ordner}'")
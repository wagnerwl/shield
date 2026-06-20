import torch
import torch.nn as nn
import subprocess
import os

# ==========================================
# 0. DEINE MODELLKLASSE DEFINIEREN
# ==========================================
class SoundDetectorCNN(nn.Module):
    def __init__(self):
        super(SoundDetectorCNN, self).__init__()
        import torch.nn.functional as F
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(5120, 64) 
        self.dropout = nn.Dropout(p=0.6) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
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
# Ersetze dies durch den echten Pfad zu deiner .pt Datei
modell_pfad = "ai_model/models/mein_geraeusch_cnn_008.pt" 

# Lädt die Gewichte (ignoriert evtl. Fehler, falls Teile fehlen, aber strict=True ist sicherer)
model.load_state_dict(torch.load(modell_pfad, map_location=torch.device('cpu')))
model.eval() 

# Der von uns berechnete Dummy-Input
dummy_input = torch.randn(1, 1, 64, 40) 
onnx_path = "Arduino/onnx/audio_cnn.onnx"

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

# onnx2tf arbeitet am besten als Kommandozeilen-Tool. 
# Wir rufen es hier direkt aus dem Python-Skript heraus auf.
ausgabe_ordner = "Arduino/tflite_ausgabe"

# Falls der Ordner nicht existiert, wird er von onnx2tf angelegt
subprocess.run([
    "onnx2tf", 
    "-i", onnx_path, 
    "-o", ausgabe_ordner,
    "--non_verbose" # Unterdrückt die gigantische Textausgabe in der Konsole
], check=True)

print(f"\n-> BÄM! Erfolgreich abgeschlossen!")
print(f"-> Dein fertiges TFLite Modell sowie das SavedModel liegen im Ordner: '{ausgabe_ordner}'")
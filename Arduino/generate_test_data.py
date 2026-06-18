import torch
import torchaudio
import torchaudio.transforms as T
import os

# ==========================================
# 1. PFAD ZUR TEST-DATEI ANPASSEN
# ==========================================
# Nimm dir am besten eine Datei aus deinem "pos" oder "neg_hard" Ordner, 
# bei der du weißt, was das Modell sagen sollte.
wav_pfad = "ai_model/data/processed/val/pos/2sec_10811_glassbreak_6641ms_normalized_16kHz_1280ms.wav" 

print(f"Lade Audio: {wav_pfad}")
waveform, sr = torchaudio.load(wav_pfad)

# ==========================================
# 2. AUDIO-UMWANDLUNG (Exakt wie in dataset.py)
# ==========================================
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)

mel_spec = mel_transform(waveform)

# Normalisierung
mel_spec_mean = mel_spec.mean()
mel_spec_std = mel_spec.std()
mel_spec = (mel_spec - mel_spec_mean) / (mel_spec_std + 1e-6) 

# Mach das 2D-Bild zu einer langen Liste
flat_data = mel_spec.flatten().numpy()

# === NEU: SICHERHEITS-KORREKTUR ===
import numpy as np
if len(flat_data) > 2560:
    flat_data = flat_data[:2560] # Zu lang? Abschneiden!
elif len(flat_data) < 2560:
    flat_data = np.pad(flat_data, (0, 2560 - len(flat_data)), 'constant') # Zu kurz? Mit Nullen auffüllen!
# ==================================

# ==========================================
# 3. EXPORT ALS C-HEADER FÜR DEN ARDUINO
# ==========================================
header_pfad = "test_data.h"
with open(header_pfad, "w") as f:
    f.write("// Automatisch generiertes Mel-Spektrogramm für Arduino\n")
    f.write("const float g_test_input[2560] = {\n")
    
    # Schreibe die Zahlen kommasepariert in die Datei
    werte = [f"{val:.6f}f" for val in flat_data]
    f.write(", ".join(werte))
    
    f.write("\n};\n")

print(f"Fertig! 2560 Werte gespeichert in: {header_pfad}")
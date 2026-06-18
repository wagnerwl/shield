import torch
import torchaudio
import torch.nn.functional as F

# 1. PFAD ZUR POSITIVEN TEST-DATEI ANPASSEN
wav_pfad = "ai_model/data/processed/val/pos/2sec_2sec_4-204115-A-39_normalized_16kHz_1280ms.wav" 
header_pfad = "Arduino/test_data/real_audio_test.h"

print(f"Lade Audio: {wav_pfad}")
waveform, sr = torchaudio.load(wav_pfad)

# Sicherstellen, dass es 1D ist (Mono)
waveform = waveform.squeeze()
# Falls das Audio fälschlicherweise 2D geblieben ist (z.B. Stereo), erzwinge Mono:
if waveform.dim() > 1:
    waveform = waveform[0] 

# ==========================================
# DEIN FIX: AUDIO VORHER EXAKT ZUSCHNEIDEN/AUFFÜLLEN
# 1.28 Sekunden * 16000 Hz = exakt 20480 Samples
# ==========================================
ziel_samples = 20480
aktuelle_samples = waveform.shape[0]

if aktuelle_samples > ziel_samples:
    print(f"Audio zu lang ({aktuelle_samples} Samples). Schneide ab auf {ziel_samples}...")
    waveform = waveform[:ziel_samples]
elif aktuelle_samples < ziel_samples:
    print(f"Audio zu kurz ({aktuelle_samples} Samples). Fülle mit Stille auf...")
    # F.pad erwartet bei 1D Tensoren ein Tuple (padding_left, padding_right)
    waveform = F.pad(waveform, (0, ziel_samples - aktuelle_samples))
else:
    print("Audio hat bereits die perfekte Länge!")

# ==========================================
# EXPORT ALS C-HEADER FÜR DEN ARDUINO
# ==========================================
with open(header_pfad, "w") as f:
    f.write("#ifndef REAL_AUDIO_TEST_H\n")
    f.write("#define REAL_AUDIO_TEST_H\n\n")
    f.write(f"// Generiert aus: {wav_pfad}\n")
    f.write(f"// Sample Rate: {sr} Hz, Länge: {len(waveform)} Samples\n\n")
    f.write(f"const float g_real_audio[{len(waveform)}] = {{\n")
    
    # Werte formatieren und schreiben
    werte = [f"{val.item():.6f}f" for val in waveform]
    
    # Zeilenumbrüche alle 10 Werte für bessere Lesbarkeit in der IDE
    for i in range(0, len(werte), 10):
        f.write(", ".join(werte[i:i+10]))
        if i + 10 < len(werte):
            f.write(",\n")
            
    f.write("\n};\n\n#endif // REAL_AUDIO_TEST_H\n")

print(f"Fertig! Exakt {len(waveform)} rohe Audio-Werte in {header_pfad} gespeichert.")
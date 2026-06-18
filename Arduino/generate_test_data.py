import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

# 1. PFAD ZUR POSITIVEN TEST-DATEI ANPASSEN
# Nimm dir am besten eine Datei aus deinem "pos" oder "neg_hard" Ordner, 
# bei der du weißt, was das Modell sagen sollte.
wav_pfad = "ai_model/data/processed/val/pos/2sec_2sec_4-204115-A-39_normalized_16kHz_1280ms.wav" 

print(f"Lade Audio: {wav_pfad}")
waveform, sr = torchaudio.load(wav_pfad)

# ==========================================
# DER FIX: AUDIO VORHER EXAKT ZUSCHNEIDEN
# 1.28 Sekunden * 16000 Hz = exakt 20480 Samples
# ==========================================
ziel_samples = 20480

if waveform.shape[1] > ziel_samples:
    # Zu lang? Schneide das rohe Audio ab (nicht das Spektrogramm!)
    waveform = waveform[:, :ziel_samples]
elif waveform.shape[1] < ziel_samples:
    # Zu kurz? Fülle das Audio mit Stille (Nullen) auf
    waveform = F.pad(waveform, (0, ziel_samples - waveform.shape[1]))

# ==========================================
# 2. AUDIO-UMWANDLUNG 
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

# ==========================================
# DER ULTIMATIVE MATRIX-FIX
# Wir zwingen das 2D-Bild exakt auf 40 Spalten (Zeit-Frames), 
# bevor wir es "platt" machen. So bleibt das Bild intakt!
# ==========================================
if mel_spec.shape[2] > 40:
    mel_spec = mel_spec[:, :, :40]  # Schneidet überschüssige Spalten rechts ab
elif mel_spec.shape[2] < 40:
    import torch.nn.functional as F
    mel_spec = F.pad(mel_spec, (0, 40 - mel_spec.shape[2])) # Füllt fehlende Spalten auf

# Jetzt hat das Spektrogramm zu 100 % die Form (1, 64, 40).
# Beim Flatten entstehen nun garantiert exakt 2560 Werte!
flat_data = mel_spec.flatten().numpy()

# ==========================================
# 3. EXPORT ALS C-HEADER
# ==========================================
header_pfad = "Arduino/test_data/test_data.h"
with open(header_pfad, "w") as f:
    f.write("// Automatisch generiertes Mel-Spektrogramm für Arduino\n")
    f.write("const float g_test_input[2560] = {\n")
    werte = [f"{val:.6f}f" for val in flat_data]
    f.write(", ".join(werte))
    f.write("\n};\n")

print(f"Fertig! Exakt {len(flat_data)} Werte gespeichert.")
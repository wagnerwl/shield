# src/inference_live.py
import os
import torch
import torchaudio.transforms as T
import yaml
import pyaudio
import numpy as np

# Eigene Module importieren
from model import SoundDetectorCNN

# --- 1. Kugelsichere Pfade ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
config_pfad = os.path.join(SRC_DIR, "config.yml")

# Config laden
with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

def lade_modell(modell_pfad):
    """Lädt das trainierte CNN-Modell auf die CPU."""
    print(f"Lade Modell von: {modell_pfad}")
    modell = SoundDetectorCNN()
    modell.load_state_dict(torch.load(modell_pfad, map_location=torch.device('cpu')))
    modell.eval() # Test-Modus
    return modell

def main():
    # 1. Modell laden
    modell_pfad = os.path.join(PROJECT_ROOT, "models", "mein_geraeusch_cnn_007.pt")
    if not os.path.exists(modell_pfad):
        print(f"Fehler: Modell nicht gefunden unter {modell_pfad}")
        return
        
    modell = lade_modell(modell_pfad)
    threshold = 0.6  # Ab 60% Sicherheit schlagen wir Alarm

    # 2. Audio-Parameter aus Config
    RATE = config['audio']['sample_rate']
    CLIP_LEN = config['audio']['clip_length_seconds']
    BUFFER_SIZE = int(RATE * CLIP_LEN)  # Exakt 20.480 Samples (1,28 Sekunden)
    
    # Wir lesen das Mikrofon in kleinen Chunks aus (z.B. ca. 0,3 Sekunden)
    # Das sorgt für ein flüssiges, überlappendes "Sliding Window" in Echtzeit.
    CHUNK_SIZE = 4096 
    
    # Spektrogramm-Wandler vorbereiten (muss identisch zum Training sein!)
    mel_transform = T.MelSpectrogram(
        sample_rate=RATE,
        n_mels=config['audio']['n_mels'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length']
    )

    # Unser "rollender" Audio-Speicher (startet mit Stille)
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

    # 3. PyAudio initialisieren
    p = pyaudio.PyAudio()
    
    print("\n" + "="*50)
    print("🎤 MIKROFON IST SCHARFGESCHALTET 🎤")
    print("Lausche auf dein Geräusch... (Abbruch mit CTRL+C)")
    print("="*50 + "\n")

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    try:
        with torch.no_grad():
            while True:
                # A. Daten vom Mikrofon lesen
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # B. Bytes in Numpy-Array umwandeln (16-bit PCM)
                # WICHTIG: Teilen durch 32768.0 normalisiert das Audio auf Werte zwischen -1.0 und 1.0!
                # torchaudio.load() macht das beim Training automatisch, beim Live-Mikrofon müssen wir das selbst tun.
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                # C. Den rollenden Speicher aktualisieren
                # Schiebt alte Daten nach links raus und packt neue rechts dran
                audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
                audio_buffer[-CHUNK_SIZE:] = audio_chunk

                # D. Für PyTorch vorbereiten
                tensor_data = torch.from_numpy(audio_buffer).unsqueeze(0)

                # E. Spektrogramm berechnen
                mel_spec = mel_transform(tensor_data)
                
                # ==========================================
                # NEU: Normalisierung (Exakt wie im Training!)
                # ==========================================
                mel_spec_mean = mel_spec.mean()
                mel_spec_std = mel_spec.std()
                mel_spec = (mel_spec - mel_spec_mean) / (mel_spec_std + 1e-6)
                # ==========================================
                
                # F. Batch-Dimension hinzufügen -> Shape: [1, 1, 64, 41]
                mel_spec_batch = mel_spec.unsqueeze(0)
                
                # G. Vorhersage machen
                vorhersage = modell(mel_spec_batch).item()

                # H. Ergebnis ausgeben
                if vorhersage >= threshold:
                    print(f"\r🚨 GERÄUSCH ERKANNT! (Sicherheit: {vorhersage*100:5.1f}%)" + " "*20, flush=True)
                else:
                    # Druckt einen sich aktualisierenden Balken in dieselbe Terminal-Zeile
                    bar_len = 30
                    filled = int(bar_len * vorhersage)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    print(f"\rLausche... [{bar}] {vorhersage*100:5.1f}%", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nBeende Live-Inference...")
    finally:
        # Alles sauber schließen
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
# src/inference.py
import os
import torch
import torchaudio
import torchaudio.transforms as T
import yaml

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
    """Lädt das trainierte CNN-Modell."""
    print(f"Lade Modell von: {modell_pfad}")
    modell = SoundDetectorCNN()
    
    # map_location='cpu' stellt sicher, dass es auch ohne Grafikkarte läuft
    modell.load_state_dict(torch.load(modell_pfad, map_location=torch.device('cpu')))
    
    modell.eval() # Wichtig: In den Test-Modus versetzen (kein Training mehr)
    return modell

def analysiere_audio(audio_pfad, modell, threshold=0.5):
    """Schiebt ein Sliding-Window über das Audio und macht Vorhersagen."""
    print(f"\nAnalysiere Datei: {audio_pfad} ...\n" + "-"*40)
    
    # 1. Audio laden und vorbereiten
    waveform, sr = torchaudio.load(audio_pfad)
    
    if sr != config['audio']['sample_rate']:
        resampler = T.Resample(sr, config['audio']['sample_rate'])
        waveform = resampler(waveform)
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Spektrogramm-Wandler vorbereiten (Exakt wie im Training!)
    mel_transform = T.MelSpectrogram(
        sample_rate=config['audio']['sample_rate'],
        n_mels=config['audio']['n_mels'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length']
    )
    
    # 3. Sliding Window Logik
    chunk_samples = int(config['audio']['sample_rate'] * config['audio']['clip_length_seconds'])
    # Wir rücken das Fenster immer um eine halbe Länge weiter (50% Überlappung)
    # Das verhindert, dass ein Geräusch genau auf der "Schnittkante" in zwei Hälften geteilt wird.
    step_samples = chunk_samples // 2 
    
    treffer_gefunden = False

    with torch.no_grad(): # Keine Gradienten berechnen -> spart RAM und ist schneller
        for start in range(0, waveform.shape[1] - chunk_samples + 1, step_samples):
            end = start + chunk_samples
            chunk = waveform[:, start:end]
            
            # Umwandeln in Melspektrogramm (Shape: [1, 64, 41])
            mel_spec = mel_transform(chunk)
                
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
            
            zeit_in_sekunden = start / config['audio']['sample_rate']
            
            if vorhersage >= threshold:
                treffer_gefunden = True
                print(f"[{zeit_in_sekunden:05.1f}s] 🚨 GERÄUSCH ERKANNT! (Sicherheit: {vorhersage*100:5.1f}%)")
            else:
                # Optional: Einkommentieren, wenn du auch die negativen Ausgaben sehen willst
                # print(f"[{zeit_in_sekunden:05.1f}s] Hintergrund            (Sicherheit: {vorhersage*100:5.1f}%)")
                pass

    if not treffer_gefunden:
        print("Kein Zielgeräusch in dieser Datei gefunden.")
    print("-" * 40)


def main():
    # 1. Pfad zu deinem fertigen Modell
    modell_pfad = os.path.join(PROJECT_ROOT, "models", "mein_geraeusch_cnn_007.pt")
    
    # 2. Pfad zu einer beliebigen Test-Datei 
    # Nimm hier am besten eine ungeschnittene Datei, z.B. einen Ausschnitt aus 
    # deinem Validierungs-Hintergrund, oder nimm selbst ein kurzes Audio mit dem Handy auf.

    TEST_AUDIO = "ai_model/data/processed/val/pos/"

    test_audio_pfad = os.path.join(TEST_AUDIO, "2sec_2sec_4-204115-A-39_normalized_16kHz_1280ms.wav")
    
    if not os.path.exists(test_audio_pfad):
        print(f"Test-Audio nicht gefunden unter: {test_audio_pfad}")
        print("Bitte trage den Pfad zu einer echten Datei in der main() Funktion ein.")
        return

    # Modell laden
    modell = lade_modell(modell_pfad)
    
    # Audio analysieren (Ab einer Sicherheit von 50% = 0.5)
    # Wenn du zu viele False Positives hast, setze den Threshold z.B. auf 0.8 oder 0.9!
    analysiere_audio(test_audio_pfad, modell, threshold=0.5)

if __name__ == "__main__":
    main()
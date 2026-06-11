# src/data_prep.py
import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import yaml

# Wo liegt dieses Skript? (das ist der 'src' Ordner)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Wo ist das Hauptprojekt? (Einen Ordner hoch von 'src', also 'ai_model')
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Lade die Konfiguration, damit wir die exakte Länge nutzen
config_pfad = os.path.join(SRC_DIR, "config.yml")
with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

SAMPLE_RATE = config['audio']['sample_rate']
CLIP_LENGTH_SECONDS = config['audio']['clip_length_seconds']
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CLIP_LENGTH_SECONDS) # 16000 * 1.28 = 20480

def prozessiere_lange_audiospur(input_pfad, output_ordner, datei_prefix):
    """
    Lädt eine lange Audiospur, wandelt sie in Mono/16kHz um und zerschneidet sie 
    in exakt gleich lange Häppchen.
    """
    print(f"\nVerarbeite Datei: {input_pfad} ...")
    
    if not os.path.exists(input_pfad):
        print(f"FEHLER: Datei {input_pfad} nicht gefunden!")
        return

    # Ordner erstellen, falls nicht vorhanden
    os.makedirs(output_ordner, exist_ok=True)

    # 1. Audio laden
    waveform, sr = torchaudio.load(input_pfad)

    # 2. Auf 16kHz resampeln, falls nötig
    if sr != SAMPLE_RATE:
        print(f" Resample von {sr}Hz auf {SAMPLE_RATE}Hz...")
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # 3. Auf Mono heruntermischen, falls es Stereo ist (mehrere Kanäle)
    if waveform.shape[0] > 1:
        print(" Wandle Stereo in Mono um...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. In Häppchen zerschneiden
    total_samples = waveform.shape[1]
    num_chunks = total_samples // SAMPLES_PER_CHUNK
    print(f" Schneide in {num_chunks} Chunks à {CLIP_LENGTH_SECONDS} Sekunden...")

    for i in tqdm(range(num_chunks), desc="Speichere Chunks"):
        start = i * SAMPLES_PER_CHUNK
        end = start + SAMPLES_PER_CHUNK
        
        # Den Chunk extrahieren
        chunk = waveform[:, start:end]
        
        # Speichern
        out_pfad = os.path.join(output_ordner, f"{datei_prefix}_{i:05d}.wav")
        torchaudio.save(out_pfad, chunk, SAMPLE_RATE)

def main():
    # --- 1. Definiere deine Eingangs-Dateien ---
    # os.path.join baut die Pfade automatisch richtig zusammen, egal ob Mac oder Windows
    pfad_3h_hintergrund = os.path.join(PROJECT_ROOT, "data", "raw", "background", "background_3h_features.wav")
    pfad_1h_validierung = os.path.join(PROJECT_ROOT, "data", "raw", "background", "validation_1h_features.wav")
    pfad_hard_negatives = os.path.join(PROJECT_ROOT, "data", "raw", "hard_negatives", "dev_hard_negatives_dense.wav")

    # --- 2. Definiere deine Ausgangs-Ordner ---
    ordner_train_neg = os.path.join(PROJECT_ROOT, "data", "processed", "train", "neg")
    ordner_val_neg   = os.path.join(PROJECT_ROOT, "data", "processed", "val", "neg")

    # --- 3. Lass die Maschine arbeiten ---
    # Hintergrund für Training
    prozessiere_lange_audiospur(
        input_pfad=pfad_3h_hintergrund, 
        output_ordner=ordner_train_neg, 
        datei_prefix="bg_train"
    )

    # Hard Negatives für Training (kommen in den gleichen 'neg'-Ordner!)
    prozessiere_lange_audiospur(
        input_pfad=pfad_hard_negatives, 
        output_ordner=ordner_train_neg, 
        datei_prefix="hardneg_train"
    )

    # Hintergrund für Validierung
    prozessiere_lange_audiospur(
        input_pfad=pfad_1h_validierung, 
        output_ordner=ordner_val_neg, 
        datei_prefix="bg_val"
    )

    print("\nFertig! Alle negativen Daten sind vorbereitet.")

if __name__ == "__main__":
    main()
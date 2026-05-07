import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

# --- 1. Konfiguration ---
TRAIN_DIR = 'Data_Frame/data/Artificial_DataFrame/mixes_train'
EVAL_DIR = 'Data_Frame/data/Artificial_DataFrame/mixes_eval'
OUTPUT_DIR = 'Data_Frame/data/Artificial_DataFrame/glassbreak_events'

# Zielordner erstellen, falls er nicht existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Metadaten einlesen und kombinieren ---
print("Lese CSV-Dateien ein...")
df_train = pd.read_csv('Data_Frame/data/Artificial_DataFrame/meta_clip_insertions_train.csv')
df_train['source_folder'] = TRAIN_DIR

df_eval = pd.read_csv('Data_Frame/data/Artificial_DataFrame/meta_clip_insertions_eval.csv')
df_eval['source_folder'] = EVAL_DIR

# Beide DataFrames zusammenführen
df_all = pd.concat([df_train, df_eval], ignore_index=True)

# Ausschließlich nach "glassbreak" filtern
df_glass = df_all[df_all['event'] == 'glassbreak']
print(f"Gefundene 'glassbreak'-Events insgesamt: {len(df_glass)}")

# --- 3. Audio extrahieren ---
processed_count = 0

for index, row in df_glass.iterrows():
    track_id = row['trackID']
    folder = row['source_folder']
    start_ms = row['event_start']
    end_ms = row['event_end']
    
    # Pfade bauen
    audio_path = os.path.join(folder, f"{track_id}.wav")
    out_filename = f"{track_id}_glassbreak_{start_ms}ms.wav"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    if not os.path.exists(audio_path):
        print(f"  [Warnung] Audiodatei nicht gefunden: {audio_path}")
        continue
        
    # Audio laden (sr=None behält die originale Sample Rate bei)
    y, sr = librosa.load(audio_path, sr=None)
    
    # --- 4. Zeitstempel in Samples umrechnen ---
    start_sample = int((start_ms / 1000.0) * sr)
    end_sample = int((end_ms / 1000.0) * sr)
    
    # Grenzen absichern
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)
    
    # --- 5. Transienten finden ---
    # Wir suchen den lautesten Punkt (max Amplitude) innerhalb des Event-Fensters
    event_audio = y[start_sample:end_sample]
    
    if len(event_audio) == 0:
        print(f"  [Warnung] Event-Fenster leer für Track {track_id}. Überspringe.")
        continue
        
    transient_offset = np.argmax(np.abs(event_audio))
    transient_sample = start_sample + transient_offset
    
    # --- 6. Den 2-Sekunden-Clip berechnen ---
    # Transient soll bei 20% liegen -> 0,4 Sekunden vor dem Transienten, 1,6 Sekunden danach
    pre_transient_samples = int(0.4 * sr)
    post_transient_samples = int(1.6 * sr)
    total_clip_samples = pre_transient_samples + post_transient_samples
    
    clip_start = transient_sample - pre_transient_samples
    clip_end = transient_sample + post_transient_samples
    
    # Leeres Array für den 2-Sekunden-Clip erstellen (mit Nullen gefüllt = Stille)
    clip = np.zeros(total_clip_samples)
    
    # --- 7. Audio sicher in den Clip einfügen (Zero-Padding für Ränder) ---
    # Bestimme, welcher Teil des Original-Audios gelesen werden kann
    read_start = max(0, clip_start)
    read_end = min(len(y), clip_end)
    
    # Bestimme, wo dieses Stück in das leere "clip"-Array eingefügt werden muss
    write_start = max(0, -clip_start)
    write_end = write_start + (read_end - read_start)
    
    # Audio-Ausschnitt übertragen
    clip[write_start:write_end] = y[read_start:read_end]
    
    # --- 8. Clip speichern ---
    sf.write(out_path, clip, sr)
    processed_count += 1

print(f"\nErfolgreich abgeschlossen! {processed_count} Clips wurden im Ordner '{OUTPUT_DIR}' gespeichert.")
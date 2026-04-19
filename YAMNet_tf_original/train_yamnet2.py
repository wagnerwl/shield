import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import librosa
import numpy as np
import os
import random

os.environ["TFHUB_CACHE_DIR"] = "./tfhub_cache_neu"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ---------------------------------------------------------
# 1. DEINE PFADE UND DATEINAMEN (Hier anpassen!)
# ---------------------------------------------------------
CSV_GOOD = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/metadata_glass.csv"           # Deine CSV für die positiven Samples
CSV_BAD = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/metadata_negatives.csv"       # Deine CSV für die negativen Samples

DIR_GOOD = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/positive_samples"   # Ordner mit den positiven .wav Dateien
DIR_BAD = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/negative_samples"     # Ordner mit den negativen .wav Dateien

# ---------------------------------------------------------
# 2. DATEINAMEN AUS DEN CSVS AUSLESEN
# ---------------------------------------------------------
print("Lese CSV-Dateien...")
df_good = pd.read_csv(CSV_GOOD)
df_bad = pd.read_csv(CSV_BAD)

# Wir nehmen ca. 50 Dateien aus jedem Datensatz für die Kalibrierung.
# Falls deine Spalte in der CSV nicht 'fname' heißt, ändere es hier entsprechend.
good_filenames = df_good['fname'].head(50).astype(str).tolist()
bad_filenames = df_bad['fname'].head(50).astype(str).tolist()

# ---------------------------------------------------------
# 3. VOLLE PFADE ZU DEN AUDIODATEIEN BASTELN
# ---------------------------------------------------------
calibration_files = []

# Gute Dateien verknüpfen
for fname in good_filenames:
    if not fname.endswith('.wav'):
        fname += '.wav'
    calibration_files.append(os.path.join(DIR_GOOD, fname))

# Schlechte Dateien verknüpfen
for fname in bad_filenames:
    if not fname.endswith('.wav'):
        fname += '.wav'
    calibration_files.append(os.path.join(DIR_BAD, fname))

# Die Liste gut durchmischen, damit der Konverter positiv/negativ abwechselnd sieht
random.shuffle(calibration_files)
print(f"Verwende {len(calibration_files)} Dateien zur Kalibrierung.")

# ---------------------------------------------------------
# 4. GENERATOR FÜR DIE AUDIODATEN
# ---------------------------------------------------------
def representative_dataset_gen():
    for filepath in calibration_files:
        if not os.path.exists(filepath):
            print(f"Überspringe (Datei nicht gefunden): {filepath}")
            continue
            
        try:
            # Lade Audio: YAMNet braucht zwingend Mono und 16.000 Hz
            waveform, _ = librosa.load(filepath, sr=16000, mono=True)
            
            # Umwandeln in das von TFLite erwartete Float32 Array
            waveform = np.array(waveform, dtype=np.float32)
            
            yield [waveform]
        except Exception as e:
            print(f"Warnung: Konnte {filepath} nicht laden: {e}")

# ---------------------------------------------------------
# 5. YAMNET LADEN & KONVERTIEREN (OHNE KERAS!)
# ---------------------------------------------------------
print("Lade Original YAMNet...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'

# hub.resolve lädt das Modell herunter und gibt uns direkt den Ordnerpfad 
# zum fertigen TensorFlow SavedModel zurück. Kein Keras nötig!
model_dir = hub.resolve(yamnet_model_handle)

print("Richte Konverter für Int8-Quantisierung ein...")
# Wir schieben diesen Ordner direkt in den Konverter
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

# Standard-Optimierung aktivieren
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Unseren Generator mit den echten Daten übergeben
converter.representative_dataset = representative_dataset_gen

# WICHTIG FÜR MIKROCONTROLLER: Alles strikt auf 8-Bit zwingen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Konvertiere Modell (das kann einige Minuten dauern)...")
tflite_quant_model = converter.convert()

# ---------------------------------------------------------
# 6. MODELL SPEICHERN
# ---------------------------------------------------------
output_filename = 'yamnet_glass_int8.tflite'
with open(output_filename, 'wb') as f:
    f.write(tflite_quant_model)
    
print(f"Erfolg! Das 8-Bit Modell für den Mikrocontroller wurde gespeichert unter: {output_filename}")
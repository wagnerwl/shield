import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os
import random
from scipy.io import wavfile
import scipy.signal

# --- DEINE PFADE HIER WIEDER EINTRAGEN ---
CSV_GOOD = "metadata_glass.csv"           
CSV_BAD = "metadata_background.csv"       
DIR_GOOD = "pfad/zu/deinem/ordner_good"   
DIR_BAD = "pfad/zu/deinem/ordner_bad"     

# 1. Dateipfade sammeln (wie vorher)
df_good = pd.read_csv(CSV_GOOD)
df_bad = pd.read_csv(CSV_BAD)
good_filenames = df_good['fname'].head(50).astype(str).tolist()
bad_filenames = df_bad['fname'].head(50).astype(str).tolist()

calibration_files = []
for fname in good_filenames:
    calibration_files.append(os.path.join(DIR_GOOD, fname if fname.endswith('.wav') else fname + '.wav'))
for fname in bad_filenames:
    calibration_files.append(os.path.join(DIR_BAD, fname if fname.endswith('.wav') else fname + '.wav'))
random.shuffle(calibration_files)

# 2. Exakte Vorverarbeitung aus deinem Original-Skript
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def representative_dataset_gen():
    for filepath in calibration_files:
        if not os.path.exists(filepath): continue
        try:
            sample_rate, wav_data = wavfile.read(filepath)
            
            # Stereo Handling (Rechter Kanal)
            if wav_data.ndim > 1:
                wav_data = wav_data[:, 1]
                
            # Resampling
            sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
            waveform = wav_data.astype(np.float32)
            
            # NORMALISIERUNG (Verhindert das 0.5-Clipping!)
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            waveform = np.clip(waveform, -1.0, 1.0)
            
            yield [waveform]
        except Exception as e:
            print(f"Fehler: {e}")

# 3. Modell Konvertieren
print("Lade Modell und konvertiere...")
model_dir = hub.resolve('https://tfhub.dev/google/yamnet/1')
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
with open('yamnet_glass_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print("Neues Modell gespeichert!")
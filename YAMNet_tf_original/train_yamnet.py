import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import librosa
import numpy as np
import os

# 1. Lade deine CSV-Datei
# Passe den Pfad an, falls die Datei woanders liegt
df = pd.read_csv('/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/metadata_glass.csv')

# Wir nehmen ca. 100-200 Dateien aus dem Trainings-Split für die Kalibrierung
df_calib = df[df['split'] == 'train'].head(150)

# 2. Generator für den Representative Dataset erstellen
# YAMNet erwartet als Input ein 1D-Array (Wellenform) mit 16 kHz Samplerate in Float32.
AUDIO_DIR = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/data_frame_SHIELD/positive_samples" # HIER PFAD ZU DEN AUDIODATEIEN EINTRAGEN

def representative_dataset_gen():
    for filename in df_calib['fname']:
        # Baue den vollen Pfad zur .wav Datei (z.B. "213166.wav")
        filepath = os.path.join(AUDIO_DIR, f"{filename}.wav")
        
        try:
            # Lade das Audio und erzwinge 16 kHz (YAMNet Standard)
            # librosa.load gibt die Wellenform und die Samplerate zurück
            waveform, _ = librosa.load(filepath, sr=16000, mono=True)
            
            # YAMNet erwartet die Form (N,) als Float32
            # Wir müssen es in einen Batch der Form (1, N) oder ähnliches packen,
            # je nach genauer TFLite-Eingabe-Signatur. 
            # Normalerweise reicht die reine Wellenform.
            waveform = np.array(waveform, dtype=np.float32)
            
            yield [waveform]
        except Exception as e:
            print(f"Fehler beim Laden von {filepath}: {e}")

# 3. Lade das Original-YAMNet Modell von TensorFlow Hub
print("Lade Original YAMNet...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
# Wir müssen es temporär speichern, um den TFLiteConverter von einem SavedModel zu nutzen
model = tf.keras.Sequential([
    hub.KerasLayer(yamnet_model_handle, trainable=False)
])
model.build([None]) # Build model
model.save("temp_yamnet_saved_model")

# 4. TFLite Converter einrichten
print("Richte Konverter für Int8-Quantisierung ein...")
converter = tf.lite.TFLiteConverter.from_saved_model("temp_yamnet_saved_model")

# Optimiere für Größe und Latenz
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Übergib die echten Glasbruch-Audiodaten zur Kalibrierung
converter.representative_dataset = representative_dataset_gen

# Zwinge den Konverter, ALLE Operationen in INT8 auszuführen (wichtig für Microcontroller)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Setze Ein- und Ausgabe strikt auf Int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 5. Konvertierung starten (das kann ein paar Minuten dauern)
print("Konvertiere Modell...")
tflite_quant_model = converter.convert()

# 6. Fertiges Modell speichern
output_filename = 'yamnet_glass_int8.tflite'
with open(output_filename, 'wb') as f:
    f.write(tflite_quant_model)
    
print(f"Erfolg! Quantisiertes Modell gespeichert unter: {output_filename}")
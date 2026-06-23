import tensorflow as tf
import numpy as np
import glob
import os

# Passe die Pfade ggf. an deine Ordnerstruktur an
saved_model_dir = "Arduino/tflite_ausgabe"
calib_dir = "ai_model/data/calibration_samples" # Der Ordner, den Skript 1 erstellt hat
ausgabe_datei = "Arduino/tflite_ausgabe/arduino_model_int8.tflite"

print("Lade Modell und starte Int8-Quantisierung...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Optimierung aktivieren
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ------------------------------------------------------------------
# Das Representative Dataset greift jetzt auf ECHTE Daten zu!
# ------------------------------------------------------------------
def representative_dataset():
    # Alle .npy Dateien finden, die wir vorhin exportiert haben
    npy_files = glob.glob(os.path.join(calib_dir, "*.npy"))
    
    if len(npy_files) == 0:
        print(f"FEHLER: Keine .npy Dateien im Ordner {calib_dir} gefunden!")
        return

    print(f"Kalibriere das Modell mit {len(npy_files)} echten Audio-Beispielen...")
    
    for f in npy_files:
        # Lädt das Numpy-Array (es hat bereits die perfekte Form [1, 64, 41, 1])
        data = np.load(f)
        # Zwingend als float32 übergeben
        yield [data.astype(np.float32)]

# Die Daten-Zufuhr an den Converter übergeben
converter.representative_dataset = representative_dataset

# Alles strikt in 8-Bit Integer (Ganzzahlen) zwingen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ein- und Ausgänge für den Arduino auf Int8 setzen
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Konvertieren (Das dauert jetzt ein paar Sekunden länger, 
# da TensorFlow die 100 Bilder analysiert!)
tflite_quant_model = converter.convert()

# Speichern
with open(ausgabe_datei, "wb") as f:
    f.write(tflite_quant_model)

print(f"✅ BÄM! Das Modell '{ausgabe_datei}' ist jetzt fehlerfrei auf Int8 optimiert.")
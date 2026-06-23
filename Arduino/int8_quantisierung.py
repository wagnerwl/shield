import tensorflow as tf
import numpy as np

saved_model_dir = "Arduino/tflite_ausgabe"

print("Lade Modell und starte Int8-Quantisierung...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Optimierung aktivieren (Zwingend für Verkleinerung)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ------------------------------------------------------------------
# Das Representative Dataset mit deinen normalisierten Werten
# ------------------------------------------------------------------
def representative_dataset():
    # ACHTUNG: Du musst die input_shape an dein Modell anpassen!
    # Wenn dein Mel-Spektrogramm z.B. 49 Zeitfenster und 40 Frequenzbänder hat:
    # input_shape = (1, 49, 40, 1) 
    input_shape = (1, 64, 41, 1) # <--- HIER DEINE ECHTEN WERTE EINTRAGEN
    
    for _ in range(100):
        # np.random.randn erzeugt automatisch Daten mit Mean=0 und Std=1
        # Das entspricht exakt deiner Normalisierung!
        dummy_mel_spec = np.random.randn(*input_shape).astype(np.float32)
        yield [dummy_mel_spec]

converter.representative_dataset = representative_dataset

# Alles in 8-Bit zwingen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ein- und Ausgänge für den Arduino auf Int8 setzen
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Konvertieren
tflite_quant_model = converter.convert()

# Speichern
ausgabe_datei = "Arduino/tflite_ausgabe/arduino_model_int8.tflite"
with open(ausgabe_datei, "wb") as f:
    f.write(tflite_quant_model)

print(f"Erfolg! Das Modell '{ausgabe_datei}' ist jetzt auf Int8 optimiert.")
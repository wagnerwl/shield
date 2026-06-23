import tensorflow as tf

# Wir laden das Ursprungsmodell (saved_model.pb)
saved_model_dir = "Arduino/tflite_ausgabe"

print("Lade Modell und konvertiere in Float16...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# WICHTIG: Um Float16 zu nutzen, MÜSSEN wir die Optimierung aktivieren
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Wir zwingen den Converter, Float16 anstelle von Int8 oder Float32 zu verwenden
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# Als neue Datei speichern (Sinnvollerweise mit "float16" im Namen)
ausgabe_datei = "Arduino/tflite_ausgabe/arduino_clean_model_float16.tflite"
with open(ausgabe_datei, "wb") as f:
    f.write(tflite_model)

print(f"Fertig! Nutze jetzt '{ausgabe_datei}' für den Arduino.")
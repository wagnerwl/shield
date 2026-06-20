import tensorflow as tf

# Wir laden den rohen Bauplan, den onnx2tf vorhin erstellt hat
saved_model_dir = "Arduino/tflite_ausgabe"

print("Lade Modell und entferne alle versteckten Optimierungen...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# WICHTIG: Wir verbieten TensorFlow jegliche Kompression!
converter.optimizations = []
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

# Als neue, saubere Datei speichern
ausgabe_datei = "Arduino/tflite_ausgabe/arduino_clean_model.tflite"
with open(ausgabe_datei, "wb") as f:
    f.write(tflite_model)

print(f"Fertig! Nutze jetzt '{ausgabe_datei}' für den Arduino.")
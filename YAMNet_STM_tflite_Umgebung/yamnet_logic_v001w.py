import numpy as np
import tensorflow as tf

# 1. Das TFLite Modell laden (Dein heruntergeladenes INT8-Modell)
model_path = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/Model/yamnet_e256_64x96_tl_int8.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)

# Speicher für die Berechnungen reservieren (wie RAM-Allokation auf dem Chip)
interpreter.allocate_tensors()

# 2. Details zu den Ein- und Ausgängen abfragen
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Erwartete Input-Form:", input_details[0]['shape']) 
# ST gibt an, dass das Modell Mel-Spektrogramme der Form (64, 96, 1) erwartet[cite: 7, 25].
print("Input Datentyp:", input_details[0]['dtype']) # Sollte int8 sein

# 3. Test-Daten vorbereiten (Der Mel-Spektrogramm Dummy)
# Hier baust du später die echte Audio-Vorverarbeitung mit 'librosa' ein.
# Für den ersten Test erzeugen wir einfach ein leeres Array in der exakt richtigen Form.
# Da das Modell quantisiert ist, muss der Input vom Typ INT8 sein!
dummy_spectrogram = np.zeros((1, 64, 96, 1), dtype=np.int8)

# 4. Die Daten in den Eingang des Modells schreiben
interpreter.set_tensor(input_details[0]['index'], dummy_spectrogram)

# 5. INFERENZ AUSFÜHREN (Das ist der Moment, in dem die KI rechnet)
print("Führe Modell aus...")
interpreter.invoke()

# 6. Das Ergebnis auslesen
output_data = interpreter.get_tensor(output_details[0]['index'])

# 7. Dequantisierung (WICHTIG!)
# Da das Modell intern mit ganzen Zahlen (INT8) rechnet, müssen wir das Ergebnis
# wieder in lesbare Prozentzahlen (Float) umwandeln.
scale, zero_point = output_details[0]['quantization']
if scale > 0:
    wahrscheinlichkeiten = (output_data.astype(np.float32) - zero_point) * scale
else:
    wahrscheinlichkeiten = output_data

print("Vorhersage (Rohdaten):", wahrscheinlichkeiten)
# Das Array hat jetzt 6 Werte (Deine 5 Klassen + die Unknown Class)
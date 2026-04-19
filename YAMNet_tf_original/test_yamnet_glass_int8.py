import tensorflow as tf
import librosa
import numpy as np
import urllib.request
import csv

# ---------------------------------------------------------
# 1. YAMNET KLASSENNAMEN LADEN (damit du Text statt Zahlen siehst)
# ---------------------------------------------------------
print("Lade Klassennamen...")
url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
urllib.request.urlretrieve(url, "yamnet_class_map.csv")

class_names = []
with open('yamnet_class_map.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Überschriften überspringen
    for row in reader:
        class_names.append(row[2]) # Der lesbare Name steht in der 3. Spalte

# ---------------------------------------------------------
# 2. TFLITE MODELL LADEN
# ---------------------------------------------------------
print("Lade 8-Bit TFLite Modell...")
interpreter = tf.lite.Interpreter(model_path="/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/yamnet_glass_int8.tflite")

# ---------------------------------------------------------
# 3. TEST-AUDIO LADEN (Passe den Pfad an!)
# ---------------------------------------------------------
# Nimm hier ein beliebiges Sample aus deinem "Good"-Ordner (Glasbruch)
TEST_AUDIO = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/Dataset/mixes_train/10004.wav" 
waveform, _ = librosa.load(TEST_AUDIO, sr=16000, mono=True)

# ---------------------------------------------------------
# 4. MODELL-EINGABE ANPASSEN & QUANTISIEREN (Float -> Int8)
# ---------------------------------------------------------
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# TFLite zwingen, die Eingabegröße an die Länge unserer Audiodatei anzupassen
interpreter.resize_tensor_input(input_details[0]['index'], [len(waveform)])
interpreter.allocate_tensors()

# Skalierungswerte aus dem Modell auslesen
input_scale, input_zero_point = input_details[0]['quantization']

# Die echten Float-Audiodaten in den 8-Bit-Bereich (-128 bis 127) umrechnen
if input_scale > 0:
    waveform_int8 = (waveform / input_scale) + input_zero_point
    waveform_int8 = np.clip(waveform_int8, -128, 127).astype(np.int8)
else:
    waveform_int8 = waveform.astype(np.int8)

# ---------------------------------------------------------
# 5. INFERENZ (Modell ausführen)
# ---------------------------------------------------------
print("Analysiere Audio...")
interpreter.set_tensor(input_details[0]['index'], waveform_int8)
interpreter.invoke()

# ---------------------------------------------------------
# 6. ERGEBNISSE AUSLESEN & DE-QUANTISIEREN (Int8 -> Float)
# ---------------------------------------------------------
# YAMNet hat 3 Ausgaben (Scores, Embeddings, Spectrogram). 
# Wir suchen den Output mit 521 Klassen (Scores).
score_tensor_index = output_details[0]['index']
output_scale, output_zero_point = output_details[0]['quantization']

for detail in output_details:
    if detail['shape'][-1] == 521: # 521 ist die Anzahl der YAMNet Klassen
        score_tensor_index = detail['index']
        output_scale, output_zero_point = detail['quantization']
        break

output_tensor = interpreter.get_tensor(score_tensor_index)

# Aus 8-Bit Zahlen wieder Wahrscheinlichkeiten (0.0 bis 1.0) machen
if output_scale > 0:
    scores = (output_tensor.astype(np.float32) - output_zero_point) * output_scale
else:
    scores = output_tensor

# YAMNet bewertet das Audio in Abschnitten (ca. alle 0.5 Sekunden).
# Wir bilden den Durchschnitt über alle Abschnitte.
mean_scores = np.mean(scores, axis=0)

# ---------------------------------------------------------
# 7. TOP 5 ERGEBNISSE ANZEIGEN
# ---------------------------------------------------------
top_5_indices = np.argsort(mean_scores)[::-1][:5]

print("\n--- TOP 5 ERGEBNISSE ---")
for i, index in enumerate(top_5_indices):
    print(f"{i+1}. {class_names[index]:<25} : {mean_scores[index]:.4f} (Sicherheit)")
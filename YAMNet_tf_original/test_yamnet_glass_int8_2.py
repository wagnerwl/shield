import tensorflow as tf
import librosa
import numpy as np
import urllib.request
import csv
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. YAMNET KLASSENNAMEN LADEN
# ---------------------------------------------------------
print("Lade Klassennamen...")
url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
urllib.request.urlretrieve(url, "yamnet_class_map.csv")

class_names = []
with open('yamnet_class_map.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Überschriften überspringen
    for row in reader:
        class_names.append(row[2])

# ---------------------------------------------------------
# 2. TFLITE MODELL LADEN
# ---------------------------------------------------------
print("Lade 8-Bit TFLite Modell...")
interpreter = tf.lite.Interpreter(model_path="/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/yamnet_glass_int8.tflite")

# ---------------------------------------------------------
# 3. AUDIO LADEN & VORBEREITEN
# ---------------------------------------------------------
TEST_AUDIO = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/Dataset/mixes_train/10018.wav"

# librosa übernimmt automatisch Mono-Konvertierung und 16kHz Resampling.
# Es liefert die Daten bereits korrekt skaliert für YAMNet.
waveform, sample_rate = librosa.load(TEST_AUDIO, sr=16000, mono=True)

# max_val = np.max(np.abs(waveform))
# if max_val > 0:
#     waveform = waveform / max_val
#     print(f"Audio normalisiert. Max-Wert war: {max_val:.4f}")
# waveform = np.clip(waveform, -1.0, 1.0)

# ---------------------------------------------------------
# 4. MODELL-EINGABE QUANTISIEREN (Float -> Int8)
# ---------------------------------------------------------
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]['index'], [len(waveform)])
interpreter.allocate_tensors()

input_scale, input_zero_point = input_details[0]['quantization']

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
score_tensor_index = output_details[0]['index']
output_scale, output_zero_point = output_details[0]['quantization']

for detail in output_details:
    if detail['shape'][-1] == 521: # YAMNet hat 521 Klassen
        score_tensor_index = detail['index']
        output_scale, output_zero_point = detail['quantization']
        break

output_tensor = interpreter.get_tensor(score_tensor_index)

if output_scale > 0:
    scores_np = (output_tensor.astype(np.float32) - output_zero_point) * output_scale
else:
    scores_np = output_tensor

# ---------------------------------------------------------
# 7. AUSWERTUNG (Silence entfernen & Peak finden)
# ---------------------------------------------------------
# Schritt A: "Silence" Klasse nullen
try:
    silence_idx = class_names.index('Silence')
    scores_np[:, silence_idx] = 0.0
    print("\nInfo: Klasse 'Silence' wurde ausgeblendet.")
except ValueError:
    pass

# Schritt B: Maximum über alle Frames suchen (Peak Detection)
max_scores_per_class = scores_np.max(axis=0)
top_class_index = max_scores_per_class.argmax()
infered_class = class_names[top_class_index]
top_score = max_scores_per_class[top_class_index]

print("-" * 30)
print(f'Top erkanntes Event: {infered_class}')
print(f'Sicherheit (Score):  {top_score:.2f}')
print("-" * 30)

# Top 5 Klassen anzeigen
top_5_indices = np.argsort(max_scores_per_class)[::-1][:5]
print("Top 5 erkannte Klassen (Peak):")
for i in top_5_indices:
    print(f"- {class_names[i]}: {max_scores_per_class[i]:.2f}")

# ---------------------------------------------------------
# 8. VISUALISIERUNG
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# Wir plotten die Verlaufskurven der Top 5 erkannten Klassen über die Frames
for i in top_5_indices:
    label = class_names[i]
    curve = scores_np[:, i]
    plt.plot(curve, label=label, linewidth=2)

# Anzeige schöner formatieren
dateiname = TEST_AUDIO.split('/')[-1]
plt.title(f'Erkennungswahrscheinlichkeit über die Zeit ({dateiname}) | 8-Bit TFLite')
plt.xlabel('Zeit (Frames, ca. alle 0.48s)')
plt.ylabel('Confidence Score (0-1)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
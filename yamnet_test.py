import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal
from scipy.io import wavfile
from IPython.display import Audio

# --- 1. Modell & Klassennamen laden ---
model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])
  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

# --- 2. Audio laden und Vorbereiten ---
wav_file_name = 'Dataset/mixes_eval/20023.wav' # Dein Dateipfad
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

# A. Stereo Handling: Nur einen Kanal nehmen
if wav_data.ndim > 1:
    # 0 = Linker Kanal, 1 = Rechter Kanal
    wav_data = wav_data[:, 1] 

# B. Resampling auf 16kHz
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Info anzeigen
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')

# C. Normalisierung
# Zuerst in Float umwandeln
waveform = wav_data.astype(np.float32)

# Normalisieren basierend auf dem tatsächlichen Max-Wert 
max_val = np.max(np.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val
    print(f"Audio normalisiert. Max-Wert war: {max_val}")
else:
    print("Warnung: Audio ist komplett leer (stumm).")

# Clipping auf -1.0 bis 1.0 
waveform = np.clip(waveform, -1.0, 1.0)


# --- 3. Modell ausführen ---
scores, embeddings, spectrogram = model(waveform)
scores_np = scores.numpy()


# --- 4. Auswertung (Silence entfernen & Peak finden) ---

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


# --- 5. Visualisierung ---
plt.figure(figsize=(12, 6))

# Wir plotten die Verlaufskurven der Top 5 erkannten Klassen
for i in top_5_indices[:5]:
    label = class_names[i]
    curve = scores_np[:, i]
    plt.plot(curve, label=label, linewidth=2)

plt.title(f'Erkennungswahrscheinlichkeit über die Zeit ({wav_file_name})')
plt.xlabel('Zeit (Frames)')
plt.ylabel('Confidence Score (0-1)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
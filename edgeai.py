import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
import scipy.signal
from scipy.io import wavfile

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
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

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

# wav_file_name = 'speech_whistling2.wav'
wav_file_name = 'Dataset/mixes_eval/20002.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# Falls Stereo (mehr als 1 Kanal), nimm den Durchschnitt beider Kanäle
if len(wav_data.shape) > 1:
    wav_data = wav_data[:, 1]

sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Listening to the wav file.
Audio(wav_data, rate=sample_rate)

waveform = wav_data / tf.int16.max
print(f"Maximaler Wert im Waveform-Array: {np.max(np.abs(waveform))}")
# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)
# 1. Datei laden
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
#infered_class = class_names[scores_np.mean(axis=0).argmax()]
max_scores_per_class = scores_np.max(axis=0)
top_class_index = max_scores_per_class.argmax()
infered_class = class_names[top_class_index]

print(f'Das stärkste erkannte Signal ist: {infered_class}')
print(f'Sicherheit (Score): {max_scores_per_class[top_class_index]:.2f}')
print(f'The main sound is: {infered_class}')

top_5_indices = np.argsort(max_scores_per_class)[::-1][:5]
print("\nTop 5 erkannte Klassen (Peak):")
for i in top_5_indices:
    print(f"- {class_names[i]}: {max_scores_per_class[i]:.2f}")

plt.figure(figsize=(10, 6))

try:
    silence_idx = class_names.index('Silence')
    scores_np[:, silence_idx] = 0  # Silence komplett eliminieren
    print("Klasse 'Silence' wurde für die Auswertung ignoriert.")
except ValueError:
    pass # Falls Silence nicht in der Liste ist (unwahrscheinlich)

# 2. Neues Maximum suchen (ohne Silence)
max_scores = scores_np.max(axis=0)
top_class_idx = max_scores.argmax()
infered_class = class_names[top_class_idx]

print(f'\nTop erkanntes Event (ohne Silence): {infered_class}')
print(f'Score: {max_scores[top_class_idx]:.2f}')

# 3. Top 5 noch einmal anzeigen
top_5_indices = np.argsort(max_scores)[::-1][:5]
print("\nTop 5 Events (ohne Silence):")
for i in top_5_indices:
    print(f"- {class_names[i]}: {max_scores[i]:.2f}")

# --- VISUALISIERUNG DES VERLAUFS (WICHTIG!) ---
# Damit siehst du, wann YAMNet was erkannt hat

plt.figure(figsize=(10, 6))

# Wir plotten nur die Top 3 erkannten Klassen über die Zeit
for i in top_5_indices[:3]:
    # Wir holen den Namen und die Kurve für diese Klasse
    label = class_names[i]
    curve = scores_np[:, i]
    plt.plot(curve, label=label)

plt.title('Wahrscheinlichkeit der Klassen über die Zeit')
plt.xlabel('Zeit (Frames)')
plt.ylabel('Confidence Score')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal
from scipy.io import wavfile

# --- 1. Modell & Klassennamen laden ---
model_path = 'yamnet_256_64x96_tl.h5' 
# Das .h5 Modell von ST muss über Keras geladen werden
model = tf.keras.models.load_model(model_path, compile=False)

def get_esc10_class_names():
    """
    Da .h5 Modelle keine Metadaten-Pfade speichern, definieren wir hier 
    die ESC-10 Klassen direkt (Reihenfolge gemäß ST Model Zoo).
    """
    return ['Dog', 'Rain', 'Sea waves', 'Crying baby', 'Sneezing', 
            'Clock tick', 'Person sneeze', 'Helicopter', 'Chainsaw', 'Rooster']

class_names = get_esc10_class_names()

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# --- 2. Audio laden und Vorbereiten ---
wav_file_name = 'Dataset/mixes_eval/20024.wav' 
sample_rate, wav_data = wavfile.read(wav_file_name)

if wav_data.ndim > 1:
    wav_data = wav_data[:, 0] # Standardmäßig linken Kanal

sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

waveform = wav_data.astype(np.float32)
max_val = np.max(np.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val

waveform = np.clip(waveform, -1.0, 1.0)

# --- 3. Modell ausführen ---
# WICHTIG: Die ST-Modelle erwarten oft einen Batch-Eingang und 
# geben meist NUR die Scores zurück (keine Embeddings/Spectrogram).
# Zudem muss das Audio oft in Segmente zerteilt werden, die das Modell versteht.

# Wir fügen eine Batch-Dimension hinzu [Batch, Samples]
# Falls das Modell eine feste Input-Länge braucht (z.B. 1 Sekunde), 
# schneiden wir hier den ersten Teil aus:
input_size = model.input_shape[1] 
if len(waveform) > input_size:
    waveform_input = waveform[:input_size]
else:
    waveform_input = np.pad(waveform, (0, input_size - len(waveform)))

# Inferenz
prediction = model.predict(np.expand_dims(waveform_input, axis=0))
scores_np = np.array(prediction) # Shape: (1, 10) bei ESC-10

# --- 4. Auswertung ---
# Da das .h5 Modell meist nur einen Score pro Clip liefert (statt Frame-basiert),
# nehmen wir den ersten (und einzigen) Frame.
final_scores = scores_np[0]

top_class_index = final_scores.argmax()
infered_class = class_names[top_class_index]
top_score = final_scores[top_class_index]

print("-" * 30)
print(f'Top erkanntes Event: {infered_class}')
print(f'Sicherheit (Score):  {top_score:.2f}')
print("-" * 30)

# Top 5 anzeigen
top_5_indices = np.argsort(final_scores)[::-1][:5]
print("Top 5 erkannte Klassen:")
for i in top_5_indices:
    print(f"- {class_names[i]}: {final_scores[i]:.2f}")

# --- 5. Visualisierung ---
plt.figure(figsize=(10, 5))
plt.bar(class_names, final_scores, color='skyblue')
plt.xticks(rotation=45)
plt.title(f'Erkennungsergebnis für {wav_file_name}')
plt.ylabel('Confidence Score')
plt.tight_layout()
plt.show()
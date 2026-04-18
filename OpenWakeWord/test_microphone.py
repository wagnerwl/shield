import os
import numpy as np
import pyaudio
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

# 1. Konfiguration
MODEL_PATH = "glass_break_model_v2.onnx"
THRESHOLD = 0.6  # Empfindlichkeit (0.0 bis 1.0)
CHUNK_SIZE = 1280 # 80ms bei 16kHz

# ONNX Modell laden
print(f"Lade Modell: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Feature Extraktor initialisieren
# Wir nutzen den gleichen Extraktor wie beim Training
extractor = AudioFeatures(inference_framework="onnx")

# 2. Mikrofon Setup
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

print("\n--- KI-HÖRT JETZT ZU (Strg+C zum Beenden) ---")

try:
    while True:
        # Audio-Daten vom Mikrofon lesen
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.int16)

        # Features extrahieren
        extractor(chunk)

        # Wir brauchen mindestens ein paar Frames im Puffer für eine gute Vorhersage
        # 15 Frames entsprechen etwa 1.2 Sekunden Audio
        if len(extractor.feature_buffer) >= 15:
            # Hol dir die letzten 15 Embedding-Frames
            feats = extractor.feature_buffer[-15:]
            
            # Die gleiche Logik wie im Training: Mean + Max
            avg_feat = np.mean(feats, axis=0)
            max_feat = np.max(feats, axis=0)
            combined = np.concatenate([avg_feat, max_feat]).astype(np.float32)
            combined = combined.reshape(1, 192) # Batch-Größe 1 hinzufügen

            # KI-Vorhersage berechnen
            result = session.run(None, {input_name: combined})
            probability = result[0][0][0]

            # Ergebnis anzeigen
            if probability > THRESHOLD:
                print(f"⚠️ GLASBRUCH ERKANNT! (Sicherheit: {probability:.2%})")
            else:
                # Optional: Zeige die aktuelle Wahrscheinlichkeit in einer Zeile an
                print(f"Höre zu... Pegel: {probability:.4f}", end='\r')

except KeyboardInterrupt:
    print("\nBeendet durch Nutzer.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
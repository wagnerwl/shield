import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

# Pfad zu DEINEM trainierten Modell
MODELL_PFAD = "./OpenWakeWord_v02/models/oww_fensterbruch_v06.onnx"

print("Lade Fensterbruch-Modell...")
# Wichtig: inference_framework="onnx" sagt ihm, dass er kein tflite suchen soll!
oww_model = Model(wakeword_models=[MODELL_PFAD], inference_framework="onnx")

# Audio-Setup (Mac Mikrofon)
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 Mikrofon ist an! Spiele einen Fensterbruch-Sound ab (oder klopfe laut)...")
print("Drücke STRG+C zum Beenden.\n")

try:
    while True:
        # Audio vom Mikrofon lesen
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Audio an dein Modell füttern
        prediction = oww_model.predict(audio_data)
        
        # Der Name in der Prediction entspricht deinem Modellnamen
        score = prediction.get("oww_fensterbruch_v06", 0.0)
        
        # print(f"Aktueller Score: {score*100:.1f}%")
        # Wenn die KI sich zu > 50% sicher ist, schlägt sie Alarm
        if score > 0.5:
            print(f"🚨 FENSTERBRUCH ERKANNT! (Sicherheit: {score*100:.1f}%)")
            

except KeyboardInterrupt:
    print("\nBeendet.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
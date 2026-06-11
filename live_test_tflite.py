import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

# Pfad zu DEINEM trainierten TFLite-Modell (Endung geändert)
MODELL_PFAD = "ei-detection-of-glass-breaking.5.lite"

print("Lade Fensterbruch-Modell (TFLite)...")
# Wichtig: inference_framework="tflite" sagt ihm, dass er die TFLite-Engine nutzen soll!
oww_model = Model(wakeword_models=[MODELL_PFAD], inference_framework="tflite")

# Audio-Setup (Mac Mikrofon)
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("\n🎤 Mikrofon ist an! Spiele einen Fensterbruch-Sound ab (oder klopfe laut)...")
print("Drücke STRG+C zum Beenden.\n")

consecutive_frames = 0
TRIGGER_THRESHOLD = 0.65  # Modell muss sich zu 65% sicher sein
REQUIRED_FRAMES = 1       # Modell muss 1x den Threshold knacken

try:
    while True:
        # Audio vom Mikrofon lesen
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Audio an dein Modell füttern
        prediction = oww_model.predict(audio_data)
        
        # Der Name in der Prediction entspricht normalerweise dem Dateinamen (ohne .tflite).
        # Falls es hier Probleme gibt, überprüfe mit print(prediction), wie der exakte Key heißt.
        score = prediction.get("oww_fensterbruch_v081", 0.0) 
        
        # --- LOGIK ---
        if score >= TRIGGER_THRESHOLD:
            consecutive_frames += 1  # Zähler hochsetzen
        else:
            consecutive_frames = 0   # Sofort abbrechen und nullen, wenn es nur ein kurzer Spike war!
            
        if consecutive_frames >= REQUIRED_FRAMES:
            print(f"🚨 FENSTERBRUCH ERKANNT! (Sicherheit: {score*100:.1f}% - Frames: {consecutive_frames})")
            
            # WICHTIG: Zähler nach dem Alarm zurücksetzen, damit es nicht dauerhaft spammt
            consecutive_frames = 0   
            
except KeyboardInterrupt:
    print("\nBeendet.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
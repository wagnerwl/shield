# ./Data_Frame/data/positive_samples/348106.wav
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

# --- 1. Konfiguration ---
PATH_TO_WAV = "./Data_Frame/data/positive_samples/348106.wav" # <-- HIER ANPASSEN
PATH_TO_ONNX_MODEL = "OpenWakeWord/Model/glass_break_model_v2.onnx"

# --- 2. Inferenz-Klasse ---
class GlassEvaluator:
    def __init__(self, model_path):
        print("Lade openWakeWord Audio-Encoder...")
        # Wir laden NUR den Vorverarbeiter (AudioFeatures). 
        # So sucht er gar nicht erst nach Alexa/Mycroft!
        self.preprocessor = AudioFeatures(inference_framework="onnx")
        
        print("Lade dein Glasbruch-Modell...")
        self.classifier = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
    def predict(self, wav_path):
        # Audio laden (librosa lädt als float32 zwischen -1 und 1)
        audio, _ = librosa.load(wav_path, sr=16000)
        
        # openWakeWord erwartet 80ms Fenster (1280 Samples)
        hop_size = 1280 
        scores = []
        
        print(f"Verarbeite Audiodatei: {os.path.basename(wav_path)}")
        
        for i in range(0, len(audio) - hop_size, hop_size):
            # Chunk extrahieren und von float32 in int16 konvertieren
            chunk_float = audio[i:i + hop_size]
            chunk_int16 = (chunk_float * 32767).astype(np.int16)
            
            # 1. Feature-Extraktion durch OWW (interner Puffer wird aktualisiert)
            self.preprocessor(chunk_int16)
            
            # Wir holen uns die Embeddings der letzten N Frames (Kontext-Fenster)
            # Da dein Modell beim Training den Durchschnitt über die ganze Audio-Datei gebildet hat,
            # nutzen wir hier z.B. die letzten 10 Frames (~800ms) als Kontext für den Peak
            emb = self.preprocessor.get_features(n_feature_frames=10) # Shape: (1, 10, 96)
            
            if emb.shape[1] > 0:
                # 2. Deine Features kombinieren (Mean + Max = 192)
                avg_feat = np.mean(emb[0], axis=0) 
                max_feat = np.max(emb[0], axis=0)
                combined = np.concatenate([avg_feat, max_feat]).astype(np.float32).reshape(1, 192)
                
                # 3. Klassifizierung durch dein ONNX-Modell
                result = self.classifier.run(None, {"input": combined})[0]
                scores.append(result[0][0])
            else:
                scores.append(0.0)
            
        return np.array(scores)

# --- 3. Main & Visualisierung ---
if __name__ == "__main__":
    if not os.path.exists(PATH_TO_WAV):
        print(f"Fehler: Audiodatei '{PATH_TO_WAV}' nicht gefunden!")
    elif not os.path.exists(PATH_TO_ONNX_MODEL):
        print(f"Fehler: Modell '{PATH_TO_ONNX_MODEL}' nicht gefunden!")
    else:
        evaluator = GlassEvaluator(PATH_TO_ONNX_MODEL)
        scores = evaluator.predict(PATH_TO_WAV)
        
        # Zeitachse (1280 Samples bei 16kHz = 0.08 Sekunden pro Frame)
        time_axis = np.arange(len(scores)) * 0.08

        # --- Plotten ---
        plt.figure(figsize=(12, 5))
        plt.plot(time_axis, scores, color='#2c3e50', lw=2, label='Glasbruch-Score')
        plt.fill_between(time_axis, scores, color='#3498db', alpha=0.3)
        plt.axhline(y=0.5, color='#e74c3c', linestyle='--', label='Schwellenwert (0.5)')
        
        plt.title(f'Evaluierung: {os.path.basename(PATH_TO_WAV)}', fontsize=14)
        plt.xlabel('Zeit (Sekunden)')
        plt.ylabel('Confidence Score (0-1)')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.2)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        print("-" * 30)
        print(f"Analyse beendet. Peak Score: {np.max(scores):.4f}")
        print("-" * 30)
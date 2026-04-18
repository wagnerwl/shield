import os
import numpy as np
from tqdm import tqdm
import openwakeword
from openwakeword.utils import AudioFeatures
import scipy.io.wavfile as wav

# Pfade definieren
folders = {
    "positive": "./data/positive_samples",
    "negative": "./data/negative_samples"
}

print("Initialisiere Audio-Feature-Extraktor (ONNX)...")
# Wir nutzen 1 CPU-Kern, um Konflikte auf dem Mac zu vermeiden
extractor = AudioFeatures(inference_framework="onnx", ncpu=1)

for label, folder_path in folders.items():
    print(f"\nVerarbeite Gruppe: {label}")
    output_path = f"./data/embeddings/{label}"
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for f in tqdm(files):
        audio_file = os.path.join(folder_path, f)
        
        try:
            # 1. Audio-Datei laden
            sample_rate, data = wav.read(audio_file)
            
            # 2. WICHTIG: Padding für kurze Dateien
            # Das Modell benötigt mind. 12160 Samples für das erste Embedding.
            # Wir füllen alle Dateien auf mindestens 1,5 Sekunden (24000 Samples) auf.
            min_samples = 24000 
            if len(data) < min_samples:
                # Mit Nullen (Stille) am Ende auffüllen
                data = np.pad(data, (0, min_samples - len(data)), 'constant')

            # 3. Den Extraktor für jede Datei zurücksetzen
            # (Verhindert, dass Reste vom vorherigen Clip im Puffer bleiben)
            extractor.reset()

            # 4. Audio verarbeiten
            # __call__ füttert das Audio in den internen Puffer
            extractor(data)

            # 5. Embeddings aus dem Puffer abgreifen
            # feature_buffer enthält die berechneten Vektoren
            emb = extractor.feature_buffer
            
            if emb is not None and len(emb) > 0:
                # Speichern (wir entfernen die Initialisierungs-Frames am Anfang)
                # Die ersten paar Frames sind oft Dummy-Werte, wir nehmen die echten.
                np.save(os.path.join(output_path, f.replace('.wav', '.npy')), emb)
            else:
                print(f"\n[INFO] Datei {f} war nach Padding immer noch zu kurz.")
        
        except Exception as e:
            print(f"\n[FEHLER] Datei {f}: {e}")

print("\nAlle Embeddings wurden erfolgreich gespeichert!")
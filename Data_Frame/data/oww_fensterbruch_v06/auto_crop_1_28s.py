import os
import numpy as np
from scipy.io import wavfile
import glob

# ==========================================
# 1. DEINE ORDNER-PFADE (Hier anpassen!)
# ==========================================
# Wo liegen deine aktuellen (zu langen) Glasbruch-Dateien?
INPUT_DIR = "Data_Frame/data/oww_fensterbruch_v04/positive_train" 

# Wo sollen die perfekten 1,28s Dateien gespeichert werden?
OUTPUT_DIR = "Data_Frame/data/oww_fensterbruch_v06/positive_1_28"

# ==========================================
# 2. AUDIO EINSTELLUNGEN
# ==========================================
TARGET_SR = 16000                   # OpenWakeWord nutzt immer 16 kHz
TARGET_LENGTH_SEC = 1.28            # Exaktes Modell-Sichtfenster
TARGET_SAMPLES = int(TARGET_SR * TARGET_LENGTH_SEC) # 20480 Samples
PRE_PEAK_BUFFER = int(TARGET_SR * 0.2) # 0.2 Sekunden VOR dem Knall abschneiden

def process_audio_files():
    # Zielordner erstellen, falls er nicht existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Alle .wav Dateien im Input-Ordner finden
    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    
    if not wav_files:
        print(f"Fehler: Keine .wav Dateien im Ordner '{INPUT_DIR}' gefunden!")
        return

    print(f"Starte Verarbeitung von {len(wav_files)} Dateien...")
    
    erfolgreich = 0
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        
        try:
            # Audio laden
            sr, audio = wavfile.read(file_path)
            
            # Falls Stereo, auf Mono runtermischen
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1).astype(audio.dtype)
                
            # Wir brauchen genau 16kHz
            if sr != TARGET_SR:
                print(f"Überspringe {filename}: Falsche Sample-Rate ({sr} Hz). Muss 16000 Hz sein.")
                continue
                
            # 1. Finde den lautesten Punkt (Index mit maximaler absoluter Amplitude)
            peak_idx = np.argmax(np.abs(audio))
            
            # 2. Berechne den idealen Startpunkt (0,2s VOR dem Peak)
            start_idx = peak_idx - PRE_PEAK_BUFFER
            
            # Sicherheits-Checks, damit wir nicht aus dem Array herausfallen
            if start_idx < 0:
                start_idx = 0  # Peak ist ganz am Anfang, starte bei 0
                
            end_idx = start_idx + TARGET_SAMPLES
            
            if end_idx > len(audio):
                # Die Datei ist nach hinten raus zu kurz, wir schieben das Fenster zurück
                end_idx = len(audio)
                start_idx = max(0, end_idx - TARGET_SAMPLES)
                
            # 3. Audio zuschneiden
            cropped_audio = audio[start_idx:end_idx]
            
            # Falls die Originaldatei insgesamt KÜRZER als 1,28s war, füllen wir am Ende mit Nullen (Stille) auf
            if len(cropped_audio) < TARGET_SAMPLES:
                padding = TARGET_SAMPLES - len(cropped_audio)
                cropped_audio = np.pad(cropped_audio, (0, padding), 'reflect')
                
            # 4. Neue Datei speichern
            out_path = os.path.join(OUTPUT_DIR, filename)
            wavfile.write(out_path, TARGET_SR, cropped_audio)
            
            erfolgreich += 1
            
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    print(f"\nFertig! {erfolgreich} von {len(wav_files)} Dateien erfolgreich auf exakt 1,28s zugeschnitten.")
    print(f"Du findest sie hier: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_audio_files()
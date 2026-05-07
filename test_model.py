import openwakeword
from openwakeword.model import Model
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. PFADE ANPASSEN
# ==========================================
MODELL_PFAD = "Data_Frame/data/oww_fensterbruch_v03.onnx" 
AUDIO_PFAD = "Data_Frame/data/oww_fensterbruch_v03/positive_test/2sec_2sec_163364.wav"                        
# ==========================================
   

# Wir legen genau fest, wie viel Stille (in Sekunden)
# VOR und NACH dem Geräusch eingefügt werden soll.
STILLE_VOR = 2.0  
STILLE_NACH = 2.0 
# ==========================================

def main():
    if not os.path.exists(MODELL_PFAD):
        print(f"FEHLER: Modell nicht gefunden unter {MODELL_PFAD}")
        return
    if not os.path.exists(AUDIO_PFAD):
        print(f"FEHLER: Audio-Datei nicht gefunden unter {AUDIO_PFAD}")
        return

    # 2. Modell und Audio laden
    print(f"Lade Modell: {os.path.basename(MODELL_PFAD)}...")
    oww_model = Model(wakeword_models=[MODELL_PFAD], inference_framework="onnx")

    print(f"Lade Audio: {os.path.basename(AUDIO_PFAD)}...")
    sr, audio = wav.read(AUDIO_PFAD)

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # --- 3. AUDIO IN DIE MITTE PACKEN ---
    # Wir erzeugen Nullen (absolute Stille) für davor und danach
    stille_vor_samples = int(sr * STILLE_VOR)
    stille_nach_samples = int(sr * STILLE_NACH)
    
    stille_vor = np.zeros(stille_vor_samples, dtype=audio.dtype)
    stille_nach = np.zeros(stille_nach_samples, dtype=audio.dtype)
    
    # Jetzt kleben wir alles zusammen: Stille -> Ton -> Stille
    audio_gepuffert = np.concatenate((stille_vor, audio, stille_nach))
    # ------------------------------------

    # 4. Das gepufferte Audio analysieren
    CHUNK_SIZE = 1280 
    wahrscheinlichkeiten = []
    zeitstempel = []

    print("Analysiere Audio-Stream...")
    for i in range(0, len(audio_gepuffert), CHUNK_SIZE):
        chunk = audio_gepuffert[i:i+CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)), 'constant')
            
        prediction = oww_model.predict(chunk)
        score = list(prediction.values())[0]
        
        wahrscheinlichkeiten.append(score)
        aktuelle_zeit_sekunden = (i + CHUNK_SIZE) / sr
        zeitstempel.append(aktuelle_zeit_sekunden)

    # 5. Schaubild zeichnen (Audio + Wahrscheinlichkeit)
    print("Erstelle Diagramm...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- HINTERGRUND: Die Audio-Wellenform ---
    # Die Zeitachse entspricht jetzt exakt der neuen Gesamtlänge (2s + Audio + 2s)
    zeit_audio = np.linspace(0, len(audio_gepuffert) / sr, num=len(audio_gepuffert))
    
    ax1.plot(zeit_audio, audio_gepuffert, color='lightgray', alpha=0.8, label='Audio-Signal (Wellenform)')
    ax1.set_xlabel('Zeit (in Sekunden)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Lautstärke', fontsize=12, fontweight='bold', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
    max_amp = np.max(np.abs(audio_gepuffert))
    if max_amp > 0:
        ax1.set_ylim(-max_amp * 1.3, max_amp * 1.3)

    # --- VORDERGRUND: Die Modell-Wahrscheinlichkeit ---
    ax2 = ax1.twinx()
    ax2.plot(zeitstempel, wahrscheinlichkeiten, color='#1f77b4', linewidth=3, label='Sicherheit (Glasbruch)')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Auslöse-Schwelle (50%)')
    
    ax2.set_ylabel('Sicherheit des Modells', fontsize=12, fontweight='bold', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(-0.05, 1.05)

    # --- FINISH ---
    plt.title(f'Analyse: "{os.path.basename(AUDIO_PFAD)}" (Eingebettet in Stille)', fontsize=15)
    
    # Legenden beider Achsen oben links zusammenfassen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
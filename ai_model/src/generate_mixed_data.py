# src/generate_mixed_data.py
import os
import glob
import random
import torch
import torchaudio
import yaml

# ==========================================
# --- Fixe Pfade ---
# BITTE ANPASSEN: Tausche "C:/Dein/Projekt/Pfad" gegen deinen echten Pfad aus!
# ==========================================
config_pfad = "ai_model/src/config.yml"
POS_DIR     = "ai_model/data/processed/train/pos"
BG_DIR      = "Data_Frame/data/FP_DataSet/dev/4_Alltags_Ambience"
OUTPUT_DIR  = "ai_model/data/processed/train/mixed_pos"

TARGET_TOTAL = 3000

# Config laden
with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

SAMPLE_RATE = config['audio']['sample_rate']

def load_and_resample(filepath):
    waveform, sr = torchaudio.load(filepath)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def match_length(bg_waveform, target_length):
    bg_length = bg_waveform.shape[1]
    if bg_length > target_length:
        start_idx = random.randint(0, bg_length - target_length)
        return bg_waveform[:, start_idx : start_idx + target_length]
    elif bg_length < target_length:
        repeats = (target_length // bg_length) + 1
        looped_bg = bg_waveform.repeat(1, repeats)
        return looped_bg[:, :target_length]
    return bg_waveform

def main():
    # Stellt sicher, dass der Zielordner existiert, falls noch nicht erstellt
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pos_files = glob.glob(os.path.join(POS_DIR, "*.wav"))
    bg_files = glob.glob(os.path.join(BG_DIR, "*.wav"))
    
    original_pos_files = [f for f in pos_files if "mixed_" not in os.path.basename(f)]
    current_total = len(pos_files)
    
    if not original_pos_files:
        print(f"Fehler: Keine originalen Positiv-Dateien gefunden in {POS_DIR}!")
        return
        
    if not bg_files:
        print(f"Fehler: Keine Hintergrunddateien gefunden in {BG_DIR}!")
        return

    to_generate = TARGET_TOTAL - current_total
    
    if to_generate <= 0:
        print(f"Du hast bereits {current_total} Daten. Keine neuen Daten nötig.")
        return
        
    print(f"Generiere {to_generate} neue gemischte Audiospuren...")
    
    for i in range(to_generate):
        pos_file = random.choice(original_pos_files)
        pos_audio = load_and_resample(pos_file)
        
        bg_file = random.choice(bg_files)
        bg_audio = load_and_resample(bg_file)
        
        bg_audio = match_length(bg_audio, pos_audio.shape[1])
        
        bg_gain = random.uniform(0.1, 0.7)
        bg_audio = bg_audio * bg_gain
        
        mixed_audio = pos_audio + bg_audio
        
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.9 
            
        out_name = f"mixed_{i:04d}_{os.path.basename(pos_file)}"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        torchaudio.save(out_path, mixed_audio, SAMPLE_RATE)
        
        if (i+1) % 100 == 0:
            print(f"[{i+1}/{to_generate}] Dateien generiert...")

    print("Fertig! Dein Modell hat jetzt eine massive Armee an Trainingsdaten.")

if __name__ == "__main__":
    main()
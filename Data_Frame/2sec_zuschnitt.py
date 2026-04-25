import os
from pydub import AudioSegment

def process_glass_audio(input_dir, output_dir, target_length_ms=2000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".wav", ".mp3", ".flac")):
            continue
            
        filepath = os.path.join(input_dir, filename)
        audio = AudioSegment.from_file(filepath)

        # In 16kHz, Mono, 16-bit konvertieren (Zwingend für OpenWakeWord)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Wenn kürzer als 2 Sekunden: Mit Stille auffüllen
        if len(audio) < target_length_ms:
            padding = AudioSegment.silent(duration=(target_length_ms - len(audio)))
            audio = audio + padding
            clip = audio
        else:
            # Finde den lautesten Block (den Peak)
            loudest_block = 0
            max_loudness = -float('inf')
            
            for i in range(0, len(audio) - 100, 100):
                block = audio[i:i+100]
                if block.dBFS > max_loudness:
                    max_loudness = block.dBFS
                    loudest_block = i
            
            # 500ms vor dem Peak und 1500ms danach ausschneiden
            start_time = max(0, loudest_block - 500)
            end_time = start_time + target_length_ms
            
            if end_time > len(audio):
                end_time = len(audio)
                start_time = max(0, end_time - target_length_ms)
                
            clip = audio[start_time:end_time]

        # In den neuen Ordner exportieren
        out_name = f"2sec_{filename[:-4]}.wav"
        clip.export(os.path.join(output_dir, out_name), format="wav")
        print(f"Fertig: {out_name}")

# HIER DEINE PFADE EINTRAGEN:
# 1. Deinen aktuellen Positiv-Ordner verarbeiten
process_glass_audio("Data_Frame/data/positive_samples", "Data_Frame/data/positive_2sec")

# 2. Deinen aktuellen Negativ-Ordner verarbeiten
process_glass_audio("Data_Frame/data/negative_samples", "Data_Frame/data/negative_2sec")
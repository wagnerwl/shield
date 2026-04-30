from pathlib import Path
import random
import shutil

source_dir = Path("Data_Frame/data/df_NEU/negative_2sec")
target_dir = Path("Data_Frame/data/df_NEU/negative_test")

test_ratio = 0.15  # 0.10 bis 0.20 ist erlaubt

random.seed(42)
target_dir.mkdir(parents=True, exist_ok=True)

audio_files = [
    p for p in source_dir.iterdir()
    if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac"}
]

if not audio_files:
    raise ValueError(f"Keine Audiodateien in {source_dir} gefunden.")

n_test = round(len(audio_files) * test_ratio)
n_test = max(1, n_test)
n_test = min(n_test, len(audio_files))

test_files = random.sample(audio_files, n_test)

for src in test_files:
    dst = target_dir / src.name
    shutil.move(str(src), str(dst))

print(f"{len(test_files)} Dateien verschoben von {source_dir} nach {target_dir}.")
print(f"Das sind {len(test_files) / len(audio_files):.1%} der Daten.")
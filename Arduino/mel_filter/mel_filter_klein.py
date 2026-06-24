import torchaudio.functional as F
import torch
import os

n_fft = 1024
n_mels = 64
sample_rate = 16000

print("Berechne die exakte PyTorch Mel-Filterbank (Sparse Edition)...")

mel_filters = F.melscale_fbanks(
    n_freqs=int(n_fft // 2 + 1),
    f_min=0.0,
    f_max=sample_rate / 2.0,
    n_mels=n_mels,
    sample_rate=sample_rate,
    norm=None,
    mel_scale='htk'
).T

ARDUINO_MAIN_DIR = "/Users/Jonas/Documents/Arduino/Shield_Software/main"
header_pfad = os.path.join(ARDUINO_MAIN_DIR, "mel_filters.h")

print(f"Schreibe Header nach: {header_pfad}")
os.makedirs(os.path.dirname(header_pfad), exist_ok=True)

starts, lengths, values = [], [], []

for i in range(n_mels):
    row = mel_filters[i]
    # Finde alle Werte, die größer als fast 0 sind
    nz = torch.where(row > 1e-6)[0]
    
    if len(nz) > 0:
        start_idx = nz[0].item()
        end_idx = nz[-1].item()
        length = end_idx - start_idx + 1
        
        starts.append(start_idx)
        lengths.append(length)
        
        row_vals = row[start_idx : end_idx + 1]
        values.extend([f"{v.item():.6f}f" for v in row_vals])
    else:
        starts.append(0)
        lengths.append(0)

with open(header_pfad, "w") as f:
    f.write("#ifndef MEL_FILTERS_H\n#define MEL_FILTERS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("// Automatisch generierte SPARSE Mel-Filterbank\n")
    
    f.write("const uint16_t g_mel_filter_starts[64] = {\n  " + ", ".join(map(str, starts)) + "\n};\n\n")
    f.write("const uint16_t g_mel_filter_lengths[64] = {\n  " + ", ".join(map(str, lengths)) + "\n};\n\n")
    
    f.write("const float g_mel_filter_values[] = {\n  ")
    for j in range(0, len(values), 10):
        f.write(", ".join(values[j:j+10]))
        if j + 10 < len(values): f.write(",\n  ")
    f.write("\n};\n\n#endif // MEL_FILTERS_H\n")

print(f"Erfolgreich! Die Datei '{header_pfad}' wurde erstellt.")
print(f"Ursprüngliche Werte: {n_mels * int(n_fft // 2 + 1)} (ca. 131 KB)")
print(f"Neue komprimierte Werte: {len(values)} (ca. {len(values) * 4 / 1024:.1f} KB)")
print("=> Über 100 KB Flash-Speicher gerettet!")
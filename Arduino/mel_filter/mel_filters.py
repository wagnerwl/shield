import torchaudio.functional as F
import torch

# Unsere Parameter aus der config.yml
n_fft = 1024
n_mels = 64
sample_rate = 16000

print("Berechne die exakte PyTorch Mel-Filterbank...")

# PyTorch generiert uns die Matrix. 
# Die Form ist (n_freqs, n_mels) -> also (513, 64)
mel_filters = F.melscale_fbanks(
    n_freqs=int(n_fft // 2 + 1), # Ergibt 513
    f_min=0.0,
    f_max=sample_rate / 2.0,     # Ergibt 8000.0 Hz
    n_mels=n_mels,
    sample_rate=sample_rate,
    norm=None,                   # Standard in torchaudio
    mel_scale='htk'              # Standard in torchaudio
)

# Wir drehen die Matrix um (Transponieren auf 64 x 513), 
# damit wir sie auf dem Arduino leichter in einer for-Schleife auslesen können.
mel_filters = mel_filters.T

header_pfad = "Arduino/mel_filter/mel_filters.h"

with open(header_pfad, "w") as f:
    f.write("#ifndef MEL_FILTERS_H\n")
    f.write("#define MEL_FILTERS_H\n\n")
    f.write("// Automatisch generierte Mel-Filterbank aus PyTorch\n")
    f.write(f"// Dimensionen: {n_mels} Bänder x {int(n_fft // 2 + 1)} FFT-Bins\n\n")
    
    # const sorgt dafür, dass diese ~130 KB direkt im Flash-Speicher landen 
    # und keinen RAM verbrauchen!
    f.write(f"const float g_mel_filterbank[{n_mels}][{int(n_fft // 2 + 1)}] = {{\n")
    
    for i in range(n_mels):
        f.write("  {\n    ")
        # Werte als Floats formatieren
        row_vals = [f"{val.item():.6f}f" for val in mel_filters[i]]
        
        # Zeilenumbrüche für Lesbarkeit
        for j in range(0, len(row_vals), 10):
            f.write(", ".join(row_vals[j:j+10]))
            if j + 10 < len(row_vals):
                f.write(",\n    ")
                
        if i < n_mels - 1:
            f.write("\n  },\n")
        else:
            f.write("\n  }\n")
            
    f.write("};\n\n#endif // MEL_FILTERS_H\n")

print(f"Erfolgreich! Die Datei '{header_pfad}' wurde erstellt.")
print(f"Sie enthält exakt {n_mels * int(n_fft // 2 + 1)} Gewichte für deinen Arduino.")
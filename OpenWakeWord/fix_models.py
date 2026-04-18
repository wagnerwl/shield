import os
import ssl
import urllib.request
import openwakeword

# Wir ermitteln den Pfad automatisch
base_path = openwakeword.__path__[0]
target_dir = os.path.join(base_path, "resources", "models")
os.makedirs(target_dir, exist_ok=True)

# Das sind die Links zu den Dateien, die openWakeWord zum Arbeiten braucht
urls = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
}

# SSL-Hürde auf dem Mac umgehen
context = ssl._create_unverified_context()

for name, url in urls.items():
    print(f"Lade {name} herunter...")
    dest = os.path.join(target_dir, name)
    try:
        with urllib.request.urlopen(url, context=context) as response, open(dest, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Erfolgreich gespeichert unter: {dest}")
    except Exception as e:
        print(f"Fehler bei {name}: {e}")

print("\nAlle Basis-Modelle sind nun bereit!")
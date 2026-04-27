import os
import sys

# Sag Python, dass es dein lokales OpenWakeWord Repo nutzen soll
sys.path.insert(0, "./OpenWakeWord_v02/openWakeWord")
import openwakeword as oww

# Pfad zum models-Ordner aufbauen und sicherstellen, dass er existiert
models_path = os.path.join(os.path.dirname(oww.__file__), 'resources', 'models')
os.makedirs(models_path, exist_ok=True)

print(f"Lade offizielle Modelle herunter nach: {models_path}")
oww.utils.download_models()
print("Download erfolgreich abgeschlossen!")
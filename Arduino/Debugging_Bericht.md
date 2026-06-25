# Debugging Bericht: Arduino Glasbruch-KI Pipeline

## Ziel
Wir wollten herausfinden, warum das int8-TFLite-Modell auf dem Arduino nicht die gleiche Vorhersage liefert wie auf dem PC, obwohl das Modell dort bereits funktionierte.

---

## Ausgangslage

### Symptome auf dem Arduino
Der Arduino lieferte wiederholt sehr ähnliche oder konstante Vorhersagen, z. B.:

- `Vorhersage: 0.574219`
- trotz wechselnder Audio-Situationen
- trotz scheinbar korrekter DSP-Verarbeitung

### Erste Vermutung
Es war unklar, ob das Problem in einem dieser Bereiche liegt:

- Rohaudio / Ringpuffer
- DSP-Kette / Mel-Spektrogramm
- Quantisierung
- TFLite-Modell
- TFLite Micro auf dem Arduino

---

## Projektkontext

Relevante Dateien im Projekt:

- `ai_model/src/dataset.py`
- `ai_model/src/inference_live.py`
- `ai_model/src/debug_export_spec.py`
- `Arduino/hilfskripte/test_tflite_pc.py`
- `Arduino/hilfskripte/vergleich_float32vsint8.py`
- `Arduino/tflite_ausgabe/arduino_model_int8.h`
- `Arduino/Shield_Software/main/MelSpectrogram.cpp`
- `Arduino/Shield_Software/main/ModelRunner.cpp`
- `Arduino/Shield_Software/main/main.ino`
- `Arduino/Shield_Software/main/ModelRunner.h`

---

## Was wir zuerst überprüft haben

### 1. Trainings- und Vorverarbeitungspipeline
Wir haben die Python-Seite geprüft:

- `dataset.py`
- `inference.py`
- `inference_live.py`
- `config.yml`
- `pt_to_tflite.py`

Dabei wurde bestätigt:

- `sample_rate = 16000`
- `n_fft = 1024`
- `hop_length = 512`
- `n_mels = 64`
- `clip_length_seconds = 1.28`

Die Python-Vorverarbeitung bestand aus:

- Audio laden
- Mono machen
- optional resamplen
- Mel-Spektrogramm erzeugen
- `AmplitudeToDB()`
- Z-Score-Normalisierung

---

## Arduino-DSP-Kette

### Erste Analyse
Die Arduino-DSP in `MelSpectrogram.cpp` wurde mit der Python-Seite verglichen.

### Wichtige Eigenschaften der Arduino-DSP
- FFT mit CMSIS-DSP
- Hann-Fenster
- reflektierendes Padding
- sparse Mel-Filterbank
- Log-Kompression in dB
- Top-dB-Clamping
- Z-Score-Normalisierung

### Debug-Erweiterungen
Es wurden Debug-Ausgaben ergänzt für:

- `fft_input first 10`
- `power_spectrum first 10`
- `mel_sum[0..4]`
- normalisierten Tensor
- später auch Modellinput und Modelloutput

---

## Rohdaten-Dump aus dem Arduino

### Vorgehen
Wir haben den Arduino-Ringpuffer als Rohdaten ausgelesen und in eine TXT-Datei geschrieben.

### Ziel
Die exakten 20.480 Rohsamples sollten in Python wieder eingelesen werden, um die DSP-Kette 1:1 zu vergleichen.

### Python-Helferskript
Dafür wurde `debug_export_spec.py` angepasst, sodass es:

- Rohsamples aus der TXT-Datei liest
- daraus das Spektrogramm berechnet
- die Werte ausgibt
- optional eine `.npy` speichert

---

## Der wichtigste DSP-Vergleich

### Ergebnis
Nach mehreren Vergleichen war klar:

- `fft_input`
- `power_spectrum`
- `mel_sum`
- normalisierter Tensor

waren auf Arduino und Python praktisch identisch.

### Bedeutung
Damit war die Vorverarbeitung im Wesentlichen als Fehlerquelle ausgeschlossen.

Das bedeutete:

- DSP stimmt
- Padding stimmt
- Mel-Filterbank stimmt
- dB-Normierung stimmt
- die Python- und Arduino-DSP sind praktisch gleich

---

## Modellvergleich auf PC und Arduino

### PC-Test
Mit `test_tflite_pc.py` wurde das int8-TFLite-Modell auf dem PC getestet.

#### Beispiel:
- mit einem aus dem Validierungsdatensatz umgerechneten Glasbruch-Beispiel kam etwa `0.92`
- mit einem Rohbuffer ohne Glasbruch kam etwa `0.1562`

Das zeigte:

- das PC-Modell funktioniert plausibel
- niedriger Wert bei Nicht-Glasbruch ist sinnvoll

### Arduino-Test
Der Arduino gab dagegen weiter z. B. aus:

- `Vorhersage: 0.574219`
- mehrfach konstant oder fast konstant

---

## Quantisierung und Inputvergleich

### Zusätzliche Debug-Schritte
Um sicherzugehen, dass nicht nur die DSP, sondern auch die Modell-Eingabe identisch ist, wurde der quantisierte Input verglichen.

### Ergebnis
Der quantisierte Input auf PC und Arduino war vollständig gleich.

Beispiel:

- `Quantized input first 8: [127, 111, 87, 96, 111, 116, 116, 115]`

Das war auf beiden Seiten identisch.

### Bedeutung
Damit war klar:

- Rohdaten stimmen
- DSP-Ausgabe stimmt
- quantisierter Input stimmt

Der Fehler liegt also nicht mehr in der Vorverarbeitung.

---

## Output-Vergleich

### Arduino-Output
Auf dem Arduino wurde schließlich der rohe Output untersucht:

- `Output type: 9`
- `Output scale: 0.003906`
- `Output zero point: -128`
- `Raw output int8: 19`

Das entspricht dequantisiert:

- `(19 - (-128)) * 0.003906 = 0.574219`

### PC-Output
Auf dem PC wurde mit demselben quantisierten Input geprüft:

- `Raw output tensor: [[-88]] int8`
- daraus dequantisiert:
- `0.1562`

### Schlussfolgerung
Die Dequantisierung selbst ist korrekt.

Der Unterschied entsteht **vor** der Dequantisierung, also im roh quantisierten Output des Modells.

---

## Wichtige Zusatzprüfung: Modellheader

### Behauptung geprüft
Es wurde geprüft, ob `model.h` und `arduino_model_int8.h` identisch sind.

### Ergebnis
- Hash-Abgleich durchgeführt
- Dateien sind identisch
- Modellbytes sind also nicht das Problem

### Bedeutung
Damit ist ausgeschlossen:

- falsches Modell-Header-File
- alter oder abweichender Modell-Export im Header

---

## Offline-Test auf dem Arduino

### Ziel
Zu prüfen, ob der Fehler in der Audio-/DSP-Pipeline liegt oder wirklich in der Modellinference.

### Vorgehen
Ein Offline-Test wurde gebaut:

- DSP wird umgangen
- stattdessen wird direkt ein gespeicherter quantisierter Tensor an das Modell gegeben
- der Arduino rechnet nur `Invoke()`

### Ergebnis
Trotz identischem Input bekam der Arduino weiterhin:

- `Raw output int8: 19`
- also:
- `Vorhersage: 0.574219`

### Das war der entscheidende Beweis
Damit ist klar:

- nicht DSP
- nicht Rohbuffer
- nicht Quantisierung
- nicht Input
- nicht Header-Datei

Der Fehler sitzt im **Model-Execution-Pfad auf dem Arduino**.

---

## Exakte technische Schlussfolgerung

### Was wir sicher wissen
1. Die Python- und Arduino-DSP-Kette sind praktisch identisch.
2. Der quantisierte Input ist identisch.
3. Das Modellheader-File ist identisch.
4. Der Arduino liefert trotzdem einen anderen rohen int8-Output als der PC.

### Was übrig bleibt
Damit bleiben nur noch diese realistischen Ursachen:

- TFLite Micro auf dem Arduino verhält sich anders als der Desktop-Interpreter
- ein Kernel / Operator verhält sich auf dem Arduino anders
- der Laufzeitpfad auf dem Arduino nutzt eine andere interne Modell-/Kernel-Ausführung
- ein board-/runtime-spezifischer Unterschied in der Inference

---

## Debugging-Artefakte und Hilfsdateien

### Wichtige erzeugte Dateien
- `Arduino/quant_input_arduino.txt`
- `ai_model/data/calibration_samples/arduino_dump.npy`
- `ai_model/data/calibration_samples/pc_quantized_input.txt`
- `ai_model/data/calibration_samples/offline_input.h`

### Wichtige Skript-Anpassungen
- `debug_export_spec.py`
- `test_tflite_pc.py`
- `txt_to_offline_input.py`

---

## Wichtige Codeänderungen auf dem Arduino

### `MelSpectrogram.cpp`
Debug-Ausgaben für:
- FFT-Input
- Power Spectrum
- Mel-Summen
- finalen normalisierten Tensor

### `ModelRunner.cpp`
Debug-Ausgaben für:
- Input-Scale
- Zero Point
- quantisierten Input
- Output-Typ
- Output-Scale
- Output-Zero-Point
- Raw Output int8

### `ModelRunner.h`
Es musste `#include <stdint.h>` ergänzt werden, weil `int8_t` sonst nicht bekannt war.

---

## Nebenproblem beim Build

### Fehler
Beim Kompilieren trat auf:

- `'int8_t' does not name a type`

### Ursache
In `ModelRunner.h` fehlte:

```cpp
#include <stdint.h>
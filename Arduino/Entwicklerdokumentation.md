# SHIELD - Smart Home Intelligent Edge-Level Defense
## Technische Entwickler-Dokumentation (Firmware & KI-Pipeline)

Diese Dokumentation dient als zentrale Referenz für Entwickler, um die Software-Architektur, die Signalverarbeitungskette sowie die Embedded-KI-Pipeline des SHIELD-Systems zu verstehen, zu warten und zu erweitern.

---

## 1. Systemübersicht und Einleitung

Das Projekt SHIELD (Smart Home Intelligent Edge-Level Defense) ist ein dezentrales, akustisches Überwachungssystem für Smart-Home-Umgebungen. Das primäre Ziel des Systems liegt in der performanten und fehlerfreien Erkennung spezifischer, kritischer Umgebungsgeräusche – wie beispielsweise Glasbruch – direkt am Entstehungsort (Edge-Level). Durch diesen Edge-Ansatz wird sichergestellt, dass keine permanenten Audiostreams in eine Cloud geladen werden müssen, was sowohl die Privatsphäre schützt als auch die Netzwerklast minimiert.

Die gesamte Erkennungs- und Entscheidungslogik ist hardwareseitig auf dem Mikrokontroller Arduino Portenta H7 (M7-Core) implementiert. Das System läuft in einer kontinuierlichen Echtzeitschleife, die Audiodaten lokal akquiriert, mittels digitaler Signalverarbeitung (DSP) in zeitfrequente Spektrogramme transformiert und über ein eingebettetes, quantisiertes neuronales Netzwerk klassifiziert. Erst bei einer verifizierten Erkennung wird ein digitaler Alarm über das lokale Netzwerk abgesetzt.

---

## 2. Toolchain und Abhängigkeiten

Für das Kompilieren, Modifizieren und Flashen der Firmware wird folgende Entwicklungsumgebung vorausgesetzt:

* **Entwicklungsumgebung:** Arduino IDE (Version 2.x empfohlen).
* **Board-Verwalter:** Installieren Sie über den integrierten Board-Manager das Paket "Arduino Mbed OS Boards" und wählen Sie als Zielboard den *Arduino Portenta H7 (M7-Core)* aus.
* **Hardware-Bibliotheken:** Für die digitale Audio-Erfassung über das On-Board-Mikrofon ist die offizielle PDM-Bibliothek zwingend erforderlich.
* **Digitale Signalverarbeitung (DSP):** Die Transformationen basieren auf der ARM CMSIS-DSP-Bibliothek (`arm_math.h`), welche für Cortex-M-Prozessoren hardwarebeschleunigte mathematische Operationen bereitstellt.
* **KI-Inferenz-Engine:** Zur Ausführung des neuronalen Netzes auf dem Mikrokontroller wird die Bibliothek *TensorFlow Lite for Microcontrollers* (bzw. das entsprechende *Chirale_TensorFlowLite* Inkludat) benötigt.
* **Netzwerk & Kommunikation:** Für den Verbindungsaufbau und den Datentransport werden die Bibliotheken `WiFi.h` und `PubSubClient.h` (MQTT) eingesetzt.

---

## 3. Software-Architektur und Kern-Pipeline

Die Firmware ist streng modular aufgebaut. Die zentrale Steuerungslogik befindet sich in der Hauptdatei `main.ino`, welche die Peripherie initialisiert und die zeitkritische Verarbeitungsschleife (`loop`) koordiniert. Die Pipeline ist deterministisch strukturiert und unterteilt sich in vier aufeinanderfolgende Software-Artefakte:

### 3.1 Audio-Erfassung (AudioProvider)
Die Komponente `AudioProvider.cpp` steuert das MEMS-Mikrofon über die PDM-Schnittstelle an. 
* **Speicher-Optimierung:** Um wertvollen RAM-Speicher auf dem Mikrokontroller zu sparen (ca. 41 KB Ersparnis), werden die eintreffenden PDM-Rohdaten im Interrupt-Service-Routine-Handler (`onPDMdata`) als 16-Bit-Ganzzahlen (`short`) in einem Ringpuffer abgelegt.
* **On-Demand-Konvertierung:** Erst wenn der Puffer für das definierte Zeitfenster vollständig gefüllt ist, werden die Daten blockweise in 32-Bit-Fließkommazahlen (`float`) konvertiert, mit dem Verstärkungsfaktor (`MIC_GAIN`) multipliziert und für die Inferenz bereitgestellt.
* **Signalbereinigung:** Vor der Weitergabe extrahiert das Modul den Gleichstromanteil (DC-Offset) aus dem Signal, um Verfälschungen bei der darauffolgenden FFT zu verhindern.

### 3.2 Digitale Signalverarbeitung (MelSpectrogram)
Das Modul `MelSpectrogram.cpp` transformiert den Audio-Zeitbereichsstrom in ein Mel-Frequenz-Spektrogramm. Dieses Modul wurde präzise so implementiert, dass es das Verhalten der Python-Bibliothek PyTorch exakt abbildet:
* **Reflect-Padding:** Da in PyTorch die Option `center=True` standardmäßig gesetzt ist, simuliert die Firmware ein exaktes Spiegelungs-Padding (Reflect-Padding) an den Signalrändern.
* **CMSIS-DSP FFT:** Die Transformation in den Frequenzbereich erfolgt hardwarebeschleunigt über die RFFT-Funktionen (`arm_rfft_fast_f32`) der ARM-Bibliothek.
* **Sparse Mel-Filterbank:** Um Rechenzeit und Speicher zu minimieren, wird eine mathematisch reduzierte (Sparse) Filterbank über die Arrays `g_mel_filter_starts` und `g_mel_filter_lengths` verwendet. Es werden nur die Frequenzbereiche berechnet, in denen der jeweilige Mel-Filter ungleich Null ist.
* **Log-Kompression & Normalisierung:** Die berechneten Amplituden werden über die Dezibel-Formel `10.0f * log10f(mel_sum + 1e-10f)` gestaucht. Abschließend wird eine Z-Score-Normalisierung angewendet. Wichtig hierbei ist die unvoreingenommene Varianzberechnung (Teilung durch `N-1` statt `N`), um mathematische Synchronität zu PyTorch zu garantieren.

### 3.3 KI-Inferenz (ModelRunner)
Das Modul `ModelRunner.cpp` kapselt den TensorFlow Lite Micro Interpreter.
* **Quantisierung (Float zu Int8):** Da das neuronale Netz vollständig auf INT8 quantisiert ist, konvertiert der ModelRunner die Fließkommazahlen des Spektrogramms händisch mithilfe der modellspezifischen Parameter `scale` und `zero_point` in 8-Bit-Ganzzahlen. Ein hartes Clamping begrenzt die Werte auf den Bereich von -128 bis 127.
* **Inferenz:** Der Interpreter führt das Modell innerhalb einer allokierten Speicherregion (Tensor Arena, konfiguriert auf 250 KB) aus.
* **Dequantisierung (Int8 zu Float):** Das ganzzahlige Ergebnis des Ausgangstensors wird mittels Skalierung wieder in eine standardisierte Wahrscheinlichkeit zwischen 0.0 und 1.0 transformiert.

### 3.4 Logik-Engine und Alarmierung
Überschreitet die berechnete Wahrscheinlichkeit den in der Konfiguration definierten Schwellenwert (`DETECTION_THRESHOLD`), greift die Alarmierungslogik:
* **Lokaler Alarm:** Ein digitaler Zustand wird an Pin 6 ausgegeben, welcher einen aktiven Hardware-Buzzer für 500 ms durchschaltet.
* **Netzwerk-Alarm:** Das System stößt parallel die MQTT-Übertragung an die Smart-Home-Zentrale an.
* **Puffer-Reset:** Unmittelbar nach einem Alarm wird der Audio-Ringpuffer zwangsweise geleert (`clearAudioBuffer()`). Dies verhindert eine Endlosschleife, da das System sonst sein eigenes Alarmsignal (Piepen des Buzzers) im direkt darauffolgenden Frame erneut als Glasbruch klassifizieren würde.

---

## 4. Kompilierung, Flashen und Debugging

### 4.1 Flash-Vorgang
1.  Schließen Sie den Arduino Portenta H7 über die USB-C-Schnittstelle an den Entwicklungsrechner an.
2.  Sollte das Betriebssystem oder die IDE den seriellen COM-Port des Boards nicht registrieren, muss der Mikrokontroller manuell in den Bootloader-Modus versetzt werden. Drücken Sie hierzu den Hardware-Power-Button zweimal schnell hintereinander (Double-Tap). Die On-Board-LED signalisiert den Modus durch ein grünes Pulsieren.
3.  Wählen Sie in der Arduino IDE den entsprechenden COM-Port aus und starten Sie den Kompilierungs- und Upload-Vorgang mittels "Hochladen".

### 4.2 Laufzeit-Debugging
Das System gibt kontinuierlich Telemetriedaten über die serielle Schnittstelle aus. Öffnen Sie den Seriellen Monitor mit einer gesetzten Baudrate von **115200**. Ein Standard-Durchlauf liefert folgende Performance-Metriken im Terminal:
* **Pegel (RMS):** Der aktuelle Effektivwert des Audiosignals zur Pegelüberwachung.
* **DSP-Zeit:** Die benötigte Zeit in Millisekunden für das Erstellen des Mel-Spektrogramms.
* **KI-Inference-Zeit:** Die reine Ausführungsdauer des Modells auf dem Prozessor.
* **Debug-Scanner:** Das Modul gibt permanent die minimalen und maximalen Werte sowohl im Float-Bereich als auch nach der INT8-Transformation aus. Dies erlaubt es Entwicklern, ein etwaiges Übersteuern oder Clipping in der Quantisierungskette sofort visuell zu analysieren.

---

## 5. KI-Modell Training und Pipeline (Python/PyTorch)

Das Herzstück der Erkennung ist ein dediziertes Convolutional Neural Network (CNN), welches in einer Python-Umgebung trainiert und anschließend für die Edge-Hardware aufbereitet wird.

### 5.1 Architektur des KI-Modells
Das Modell ist in `model.py` als `SoundDetectorCNN` implementiert. Es ist als flaches, aber hochgradig regularisiertes Netzwerk konzipiert, um Inferenzzeiten auf Mikrocontrollern flach zu halten. Die Schichten sind wie folgt aufgebaut:

* **Eingangsdimension:** `[Batch, 1, 64, 41]` – Repräsentiert ein Spektrogramm mit 64 Mel-Frequenzbändern und 41 Zeit-Frames (generiert aus 1.28 Sekunden Audio bei 16 kHz).
* **Faltungsblock 1 (Conv Block 1):** * `Conv2d`: 1 Eingangskanal zu 16 Ausgangskanälen (Feature Maps), Kernel-Größe 3x3, Padding=1 (erhält die Dimensionen).
    * `BatchNorm2d(16)`: Normalisiert die Feature Maps batchweise, um das Training zu stabilisieren und interne Kovarianzverschiebungen zu reduzieren.
    * `ReLU`: Aktivierungsfunktion für Nichtlinearität.
    * `MaxPool2d(2)`: Reduziert die räumliche Dimension um die Hälfte (von 64x41 auf 32x20).
* **Faltungsblock 2 (Conv Block 2):**
    * `Conv2d`: 16 Eingangskanäle zu 32 Ausgangskanälen, Kernel-Größe 3x3, Padding=1.
    * `BatchNorm2d(32)`: Normalisiert die 32 extrahierten Feature Maps.
    * `ReLU`: Aktivierungsfunktion.
    * `MaxPool2d(2)`: Erneute Halbierung der Dimensionen (von 32x20 auf 16x10).
* **Klassifikationskopf (Dense/Fully Connected Layers):**
    * `Flatten`: Transformiert die verbleibenden Feature-Maps in einen eindimensionalen Vektor der Größe 5120 (16 x 10 x 32).
    * `Linear (FC1)`: Reduziert die 5120 Features auf 64 verdeckte Neuronen, gefolgt von einer `ReLU`-Aktivierung.
    * `Dropout(p=0.6)`: Ein aggressiver Dropout friert während des Trainings zufällig 60 % der Neuronenverbindungen ein. Dies erzwingt redundante Lernpfade und schützt das Modell effektiv vor Overfitting auf die Trainingsdaten.
    * `Linear (FC2)`: Reduziert die 64 Neuronen auf 1 Ausgangsneuron.
    * `Sigmoid`: Komprimiert den finalen Netzausgang in ein Intervall zwischen 0.0 und 1.0, welches die Wahrscheinlichkeit für das Vorliegen des Zielgeräuschs repräsentiert.

### 5.2 Datensatz und Augmentation
Um eine hohe Robustheit in realen akustischen Umgebungen zu erzielen, wendet die Klasse `SoundDataset` in `dataset.py` intensive Daten-Augmentierungen während des Trainings an:
* **RIR-Faltung (Room Impulse Response):** Zufällig gewählte Impulsantworten echter Räume werden mittels FFT-Faltung (`F_audio.fftconvolve`) über das Audio-Signal gelegt. Dies simuliert Reflexionen und Hallcharakteristiken variierender Raumgrößen.
* **Lautstärke-Varianz:** Das Signal wird mit einem zufälligen Gain-Faktor zwischen 0.1 und 1.2 multipliziert.
* **SpecAugment:** Nach der Transformation in die Dezibel-Skala werden über `FrequencyMasking` (Breite: 15 Bänder) und `TimeMasking` (Breite: 10 Frames) zufällig komplette Streifen im Spektrogramm genullt. Das zwingt das CNN, verteilte Merkmale zu lernen, anstatt sich auf singuläre Frequenzen zu verlassen.

### 5.3 Trainingsprozess
* **Klassen-Balancierung:** Da negative Hintergrunddaten (`neg_bg` und `neg_hard`) die positiven Signale mengenmäßig oft übersteigen, sorgt ein `WeightedRandomSampler` im PyTorch `DataLoader` für eine gleichmäßige Verteilung der Klassen innerhalb jedes Batches.
* **Verlustfunktion (Focal Loss):** Statt Standard-BCE wird ein mathematischer Focal Loss verwendet. Dieser gewichtet mathematisch "schwierige", falsch klassifizierte Beispiele (Hard Negatives wie z. B. Besteckklirren) stark über, während bereits sicher gelernte Hintergrundgeräusche das Modell kaum noch beeinflussen.
* **Scheduler:** Ein `ReduceLROnPlateau` überwacht die Fehlalarmrate pro Stunde auf dem Validierungsset. Stagniert diese Metrik über zwei Epochen, wird die Lernrate automatisch halbiert.

---

## 6. Modell-Export und Quantisierung

Der Exportpfad von der Python-Entwicklung bis hin zur eingebetteten C++ Datei erfolgt in einer zweistufigen Toolchain:

### 6.1 PyTorch zu ONNX und SavedModel (`pt_to_tflite.py`)
Das trainierte PyTorch-Modell (`.pt`) wird über den internen Tracer in ein ONNX-Graph-Format exportiert. 
* Hierbei wird die Opset-Version 18 verwendet und die Option `do_constant_folding=True` erzwingt die Optimierung und das Einfrieren von konstanten Netzwerkparametern.
* Das CLI-Tool `onnx2tf` konvertiert die ONNX-Datei direkt in ein TensorFlow `SavedModel`. Die Flag `--keep_ncw_or_nchw_or_ncdhw_input_names input` verhindert hierbei das Vertauschen oder Invertieren der Tensor-Dimensionen (Channels First vs. Channels Last).

### 6.2 INT8-Quantisierung (`int8_quantisierung.py`)
Das Fließkomma-Modell wird über den TensorFlow Lite Converter in ein hochkomprimiertes Festkomma-Format (8-Bit Integer) transformiert:
* **Representative Dataset:** Um die Wertebereiche der Aktivierungsschichten ohne massiven Genauigkeitsverlust von Float32 auf Int8 abzubilden, lädt der Konverter echte Kalibrierungsdaten (`.npy`-Dateien aus echten Audio-Beispielen).
* **Strikte Evaluierung:** Die Konfiguration erzwingt ausschließlich INT8-Operationen (`TFLiteConverter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`).
* **I/O-Typisierung:** Sowohl der Eingangs- als auch der Ausgangstensor werden explizit als `tf.int8` definiert.

Das fertige Modell wird als `arduino_model_int8.tflite` exportiert. Dieses kann mittels des Linux-Befehls `xxd -i arduino_model_int8.tflite > model.h` in ein C-Byte-Array konvertiert werden, welches direkt vom `ModelRunner` in der Firmware eingebunden wird.

---

## 7. Netzwerk- und Home Assistant Integration

Das System kommuniziert rein lokal über das MQTT-Protokoll, um eine schnelle und unkomplizierte Integration in Home Assistant zu gewährleisten.

### 7.1 WLAN-Verbindungsaufbau
Die Firmware versucht beim Bootvorgang, sich mit den in der Konfiguration hinterlegten Zugangsdaten zu verbinden. Um ein Blockieren des Gesamtsystems bei Netzwerkausfall zu verhindern, ist eine Timeout-Logik implementiert: Das System wartet maximal 10 Sekunden (20 Versuche a 500 ms) auf die Zuweisung einer IP-Adresse. Schlägt dies fehl, bootet das System autonom im **Offline-Modus** weiter und signalisiert Alarme rein über den lokalen Buzzer.

### 7.2 MQTT-Schnittstelle (`HomeAssistant.h`)
Sobald ein gültiges Ereignis detektiert wird, baut die Funktion `sendeGlasbruchAlarm` eine Verbindung zum Broker auf:
* **Broker-Konfiguration:** IP-Adresse `134.103.184.103` auf Port `1883`.
* **Verbindungs-Robustheit:** Das System unternimmt maximal 4 sequentielle Verbindungsversuche mit einer generierten, eindeutigen Client-ID, bevor die Netzwerk-Alarmierung abgebrochen wird.
* **Daten-Payload:** Nach erfolgreichem Verbindungsaufbau wird ein kompakter JSON-String an das Topic `shield/sensor/glassbreak` publiziert:

```json
{
  "alarm": true,
  "confidence": 0.8
}
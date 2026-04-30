## Befehle:

Um die Augmentierung der Clips auszuführen kann nachdem die yml Datei konfiguriert wurde der Befehl ausgeführt werden

```bash
PYTHONPATH="./OpenWakeWord_v02/openWakeWord" python OpenWakeWord_v02/openWakeWord/openwakeword/train.py --training_config OpenWakeWord_v02/fenster_config.yml --augment_clips
```


### Training

```bash
PYTHONPATH="./OpenWakeWord_v02/openWakeWord" python OpenWakeWord_v02/openWakeWord/openwakeword/train.py --training_config OpenWakeWord_v02/fenster_config.yml --train_model
```
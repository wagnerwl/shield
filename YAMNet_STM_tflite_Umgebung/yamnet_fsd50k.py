import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Konfiguration basierend auf den FSD50K Model Zoo Spezifikationen ---
# Achte darauf, dass du das TFLite-Modell aus dem Ordner 'fsd50k/.../without_unknown_class/' nutzt
MODEL_PATH = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/Model/yamnet_e256_64x96_tl_int8.tflite" 
AUDIO_PATH = "/Users/wagner/Desktop/Uni_Reutlingen/Master HUC/IOT/Projekt/shield/Dataset/mixes_train/10018.wav" # <-- Hier deinen Pfad eintragen

# Preprocessing Parameter (identisch geblieben)
SR = 16000
N_MELS = 64
PATCH_FRAMES = 96
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 125
FMAX = 7500

# FSD50K Klassen (aus der yamnet_e256_64x96_tl_config.yaml für without_unknown_class)
CLASSES = ['Speech', 'Gunshot_and_gunfire', 'Crying_and_sobbing', 'Knock', 'Glass']

def extract_mel_spectrogram(wav_path):
    y, sr = librosa.load(wav_path, sr=SR)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window='hann', center=False, pad_mode='constant', 
        fmin=FMIN, fmax=FMAX, n_mels=N_MELS, htk=True
    )
    return np.log(mel_spectrogram + 1e-6)

def classify_audio(interpreter, mel_spectrogram):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    total_frames = mel_spectrogram.shape[1]
    predictions = []
    times = []
    
    patch_hop = int(PATCH_FRAMES * 0.75) 
    
    for start in range(0, total_frames - PATCH_FRAMES + 1, patch_hop):
        patch = mel_spectrogram[:, start:start+PATCH_FRAMES]
        patch = np.expand_dims(patch, axis=(0, -1)) 
        
        if input_scale > 0:
            patch_quant = np.clip(np.round(patch / input_scale) + input_zero_point, -128, 127).astype(np.int8)
        else:
            patch_quant = patch.astype(np.float32)
            
        interpreter.set_tensor(input_details[0]['index'], patch_quant)
        interpreter.invoke()
        
        output_quant = interpreter.get_tensor(output_details[0]['index'])[0]
        
        if output_scale > 0:
            probs = (output_quant.astype(np.float32) - output_zero_point) * output_scale
        else:
            probs = output_quant
            
        predictions.append(probs)
        time_sec = (start + PATCH_FRAMES/2) * HOP_LENGTH / SR
        times.append(time_sec)
        
    return np.array(predictions), np.array(times)

def main():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    log_mel = extract_mel_spectrogram(AUDIO_PATH)
    predictions, times = classify_audio(interpreter, log_mel)
    
    if len(predictions) == 0:
        print("Audio ist zu kurz für einen 96-Frame Patch.")
        return
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linker Plot: Spektrogramm
    img = librosa.display.specshow(log_mel, sr=SR, hop_length=HOP_LENGTH, 
                                   x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX, ax=ax[0])
    ax[0].set_title('Mel-Spektrogramm')
    fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    
    # Rechter Plot: Die 5 FSD50K Klassen
    for idx, class_name in enumerate(CLASSES):
        class_probs = predictions[:, idx]
        ax[1].plot(class_probs, times, label=class_name, marker='o')
        
    ax[1].set_title('Klassifizierung über Zeit (FSD50K Subset)')
    ax[1].set_xlabel('Wahrscheinlichkeit')
    ax[1].set_ylabel('Zeit (Sekunden)')
    ax[1].set_xlim(0, 1.05)
    ax[1].invert_yaxis() 
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
config_pfad = os.path.join(SRC_DIR, "config.yml")

with open(config_pfad, "r") as f:
    config = yaml.safe_load(f)

SAMPLE_RATE = config["audio"]["sample_rate"]
N_FFT = config["audio"]["n_fft"]
N_MELS = config["audio"]["n_mels"]
HOP_LENGTH = config["audio"]["hop_length"]
CLIP_SAMPLES = int(config["audio"]["clip_length_seconds"] * SAMPLE_RATE)
TOTAL_SAMPLES = CLIP_SAMPLES

mel_filter_bank = F.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=SAMPLE_RATE / 2.0,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    norm=None,
    mel_scale="htk",
).T  # shape: [64, 513]


def build_spec_from_waveform(waveform, sample_rate):
    if sample_rate != SAMPLE_RATE:
        waveform = T.Resample(sample_rate, SAMPLE_RATE)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]

    print("=== PY DSP DEBUG ===")
    print("waveform first 10:", waveform.flatten()[:10].tolist())

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    db_transform = T.AmplitudeToDB()

    mel_spec_raw = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec_raw)

    mel_spec = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

    print("min/max/mean/std:",
          mel_spec.min().item(),
          mel_spec.max().item(),
          mel_spec.mean().item(),
          mel_spec.std().item())

    print("first 8:", mel_spec.flatten()[:8].tolist())
    return mel_spec


def reflect_index(idx, size):
    while idx < 0 or idx >= size:
        if idx < 0:
            idx = -idx
        else:
            idx = 2 * size - 2 - idx
    return idx


def debug_first_arduino_frame(waveform):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]

    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)

    hanning_window = np.array(
        [0.5 * (1.0 - np.cos(2.0 * np.pi * i / N_FFT)) for i in range(N_FFT)],
        dtype=np.float32,
    )

    center_pos = 0
    start_index = center_pos - (N_FFT // 2)

    fft_input = np.zeros(N_FFT, dtype=np.float32)
    for i in range(N_FFT):
        pos = start_index + i
        pos_reflect = reflect_index(pos, TOTAL_SAMPLES)
        fft_input[i] = audio[pos_reflect]

    print("=== FRAME 0 DEBUG ===")
    print("fft_input first 10:", fft_input[:10].tolist())

    windowed = fft_input * hanning_window
    fft_output = np.fft.rfft(windowed, n=N_FFT)
    power_spectrum = np.abs(fft_output) ** 2
    power_spectrum = power_spectrum.astype(np.float32)

    print("power_spectrum first 10:", power_spectrum[:10].tolist())

    mel_sums = []
    value_index = 0
    for mel in range(N_MELS):
        start_bin = int(np.where(mel_filter_bank[mel].numpy() > 1e-6)[0][0])
        length = int(np.where(mel_filter_bank[mel].numpy() > 1e-6)[0][-1] - start_bin + 1)

        mel_sum = 0.0
        for k in range(length):
            mel_sum += power_spectrum[start_bin + k] * float(mel_filter_bank[mel, start_bin + k].item())
            value_index += 1

        mel_sums.append(mel_sum)

    print("mel_sum[0] = {:.6f}".format(mel_sums[0]))
    print("mel_sum[1] = {:.6f}".format(mel_sums[1]))
    print("mel_sum[2] = {:.6f}".format(mel_sums[2]))
    print("mel_sum[3] = {:.6f}".format(mel_sums[3]))
    print("mel_sum[4] = {:.6f}".format(mel_sums[4]))

    mel_db = np.array([10.0 * np.log10(x + 1e-10) for x in mel_sums], dtype=np.float32)
    print("mel_db first 5:", mel_db[:5].tolist())

    return fft_input, power_spectrum, np.array(mel_sums, dtype=np.float32), mel_db


def read_raw_audio_from_text(log_path):
    values = []
    inside = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(",", ".")
            if line == "=== RAW_AUDIO_BEGIN ===":
                inside = True
                continue
            if line == "=== RAW_AUDIO_END ===":
                break
            if inside and line:
                try:
                    values.append(float(line))
                except ValueError:
                    pass

    audio = np.array(values, dtype=np.float32)
    print(f"Geladene Samples: {len(audio)}")
    return torch.from_numpy(audio).unsqueeze(0)


if __name__ == "__main__":
    txt_path = "Arduino/RAW_Audio_serial_Output.txt"
    waveform = read_raw_audio_from_text(txt_path)

    debug_first_arduino_frame(waveform)

    spec = build_spec_from_waveform(waveform, SAMPLE_RATE)
    np.save(
        os.path.join(PROJECT_ROOT, "data", "calibration_samples", "arduino_dump.npy"),
        np.expand_dims(spec.numpy(), axis=0)
    )
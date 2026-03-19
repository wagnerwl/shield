import argparse
import math
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import stft, resample_poly


def load_audio_mono_16k(audio_path: str, target_sr: int = 16000):
    sr, x = wavfile.read(audio_path)

    if x.ndim > 1:
        x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)

    x = np.clip(x, -1.0, 1.0)

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up, down).astype(np.float32)
        sr = target_sr

    return sr, x


def hz_to_mel_htk(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz_htk(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sr, n_fft=512, n_mels=64, fmin=125.0, fmax=7500.0):
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)

    mel_min = hz_to_mel_htk(fmin)
    mel_max = hz_to_mel_htk(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz_htk(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_freqs)

        for k in range(left, center):
            fb[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / float(right - center)

    return fb


def make_patch_starts(total_frames, patch_frames, patch_hop):
    if total_frames <= patch_frames:
        return [0]
    starts = list(range(0, total_frames - patch_frames + 1, patch_hop))
    last = total_frames - patch_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def waveform_to_logmel_patches(
    waveform,
    sr=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=64,
    fmin=125.0,
    fmax=7500.0,
    patch_frames=96,
    patch_hop=48,
):
    if len(waveform) < win_length:
        waveform = np.pad(waveform, (0, win_length - len(waveform)))

    _, _, zxx = stft(
        waveform,
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )

    mag = np.abs(zxx).astype(np.float32)  # [freq_bins, time_frames]
    mel_fb = build_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = mel_fb @ mag               # [64, time_frames]
    log_mel = np.log(np.maximum(mel_spec, 1e-6)).astype(np.float32)

    if log_mel.shape[1] < patch_frames:
        pad = patch_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")

    starts = make_patch_starts(log_mel.shape[1], patch_frames, patch_hop)
    patches = [log_mel[:, s:s + patch_frames] for s in starts]
    return np.stack(patches, axis=0)      # [num_patches, 64, 96]


def quantize_input(x, input_details):
    dtype = input_details["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zp = input_details["quantization"]
        if scale <= 0:
            raise ValueError("Ungültige Input-Quantisierung: scale <= 0")
        x = np.round(x / scale + zp)
        info = np.iinfo(dtype)
        x = np.clip(x, info.min, info.max).astype(dtype)
    else:
        x = x.astype(dtype)
    return x


def dequantize_output(y, output_details):
    dtype = output_details["dtype"]
    y = y.astype(np.float32)
    if dtype in (np.int8, np.uint8):
        scale, zp = output_details["quantization"]
        if scale > 0:
            y = (y - zp) * scale
    return y


def softmax(v):
    v = v - np.max(v)
    ev = np.exp(v)
    s = np.sum(ev)
    return ev / s if s > 0 else ev


def get_class_names(num_classes):
    # STM ESC-10 Reihenfolge aus ST-Deploy-Konfig
    if num_classes == 10:
        return [
            "chainsaw",
            "clock_tick",
            "crackling_fire",
            "crying_baby",
            "dog",
            "helicopter",
            "rain",
            "rooster",
            "sea_waves",
            "sneezing",
        ]
    # STM FSD-Subset (5 Klassen)
    if num_classes == 5:
        return [
            "knock",
            "glass",
            "gunshots_and_gunfire",
            "crying_and_sobbing",
            "speech",
        ]
    return [f"class_{i}" for i in range(num_classes)]


def run_tflite(model_path, patches):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_shape = input_details["shape"]  # [1, H, W, C]
    if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[3] != 1:
        raise ValueError(f"Unerwartete Input-Shape: {in_shape}")

    in_h = int(in_shape[1])
    in_w = int(in_shape[2])

    all_scores = []

    for patch in patches:
        # patch: [64, 96]
        if patch.shape == (in_h, in_w):
            x = patch
        elif patch.shape == (in_w, in_h):
            x = patch.T
        else:
            raise ValueError(
                f"Patch-Shape {patch.shape} passt nicht zur Modell-Shape {(in_h, in_w)}"
            )

        x = x[..., np.newaxis]       # [H, W, 1]
        x = np.expand_dims(x, axis=0)  # [1, H, W, 1]
        x = quantize_input(x, input_details)

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details["index"])
        y = dequantize_output(y, output_details).reshape(-1)

        # Manche Modelle geben bereits Softmax aus, andere eher logits.
        s = np.sum(y)
        if np.any(y < 0.0) or s < 0.80 or s > 1.20:
            y = softmax(y)

        all_scores.append(y)

    return np.stack(all_scores, axis=0), input_details, output_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Model/yamnet_e256_64x96_tl_int8.tflite",
        help="Pfad zur TFLite-Datei",
    )
    parser.add_argument("--audio", type=str, required=True, help="Pfad zu WAV-Datei")
    parser.add_argument("--patch-hop", type=int, default=48, help="Patch-Schritt in Frames")
    parser.add_argument("--top-k", type=int, default=5, help="Anzahl Top-Klassen")
    args = parser.parse_args()

    sr, waveform = load_audio_mono_16k(args.audio, 16000)
    patches = waveform_to_logmel_patches(
        waveform,
        sr=sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        fmin=125.0,
        fmax=7500.0,
        patch_frames=96,
        patch_hop=args.patch_hop,
    )

    scores, in_det, out_det = run_tflite(args.model, patches)
    class_names = get_class_names(scores.shape[1])

    peak_scores = scores.max(axis=0)
    mean_scores = scores.mean(axis=0)

    top_idx = int(np.argmax(peak_scores))
    print("Model:", args.model)
    print("Audio:", args.audio)
    print("Input shape:", in_det["shape"], "dtype:", in_det["dtype"], "quant:", in_det["quantization"])
    print("Output shape:", out_det["shape"], "dtype:", out_det["dtype"], "quant:", out_det["quantization"])
    print("Num patches:", scores.shape[0])
    print()
    print("Top Event (Peak):", class_names[top_idx], f"{peak_scores[top_idx]:.4f}")
    print()

    topk = np.argsort(peak_scores)[::-1][:args.top_k]
    print("Top Klassen (Peak / Mean):")
    for i in topk:
        print(f"- {class_names[i]:24s} peak={peak_scores[i]:.4f} mean={mean_scores[i]:.4f}")


if __name__ == "__main__":
    main()
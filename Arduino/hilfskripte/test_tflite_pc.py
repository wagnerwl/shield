import numpy as np
import tensorflow as tf

MODEL_PATH = "Arduino/tflite_ausgabe/arduino_model_int8.tflite"
INPUT_PATH = "ai_model/data/calibration_samples/arduino_dump.npy"
DUMP_PATH = "ai_model/data/calibration_samples/pc_quantized_input.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"Erwartete Form (Shape): {input_details['shape']}")
print(f"Input details: {input_details}")
print(f"Output details: {output_details}")

sample = np.load(INPUT_PATH)

scale, zp = input_details["quantization"]
sample_quant = np.round(sample / scale) + zp
sample_quant = np.clip(sample_quant, -128, 127).astype(np.int8)

print("Quantized input shape:", sample_quant.shape)
print("Quantized input first 8:", sample_quant.flatten()[:8].tolist())
print("Quantized input min/max:", sample_quant.min(), sample_quant.max())

flat = sample_quant.flatten()

with open(DUMP_PATH, "w", encoding="utf-8") as f:
    f.write(f"shape: {sample_quant.shape}\n")
    f.write(f"min: {flat.min()}\n")
    f.write(f"max: {flat.max()}\n")
    f.write("values:\n")
    for value in flat:
        f.write(f"{int(value)}\n")

print(f"Quantized input written to: {DUMP_PATH}")

interpreter.set_tensor(input_details["index"], sample_quant)
interpreter.invoke()

raw_out = interpreter.get_tensor(output_details["index"])
out_scale, out_zp = output_details["quantization"]
prediction = (raw_out.astype(np.float32) - out_zp) * out_scale

print("Raw output tensor:", raw_out, raw_out.dtype)
print(f"Vorhersage des TFLite Modells am PC: {prediction[0][0]:.4f}")
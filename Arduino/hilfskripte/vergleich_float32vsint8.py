import numpy as np, tensorflow as tf

# lade Modelle
interpreter_f = tf.lite.Interpreter('Arduino/tflite_ausgabe/audio_cnn_float32.tflite'); interpreter_f.allocate_tensors()
interpreter_q = tf.lite.Interpreter('Arduino/tflite_ausgabe/arduino_model_int8.tflite'); interpreter_q.allocate_tensors()

def run_interpreter(interp, inp):
    input_details = interp.get_input_details()[0]
    # wenn Modell int8/uint8: quantisiere input entsprechend
    if input_details['dtype'] in [np.int8, np.uint8]:
        scale, zero_point = input_details['quantization']
        q_inp = np.round(inp / scale + zero_point).astype(input_details['dtype'])
        interp.set_tensor(input_details['index'], q_inp)
    else:
        interp.set_tensor(input_details['index'], inp.astype(np.float32))
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index'])
    # if output is quantized, dequantize to compare
    od = interp.get_output_details()[0]
    if od['dtype'] in [np.int8, np.uint8]:
        oscale, ozp = od['quantization']
        out = (out.astype(np.float32) - ozp) * oscale
    return out

# teste auf einer Stichprobe
x = ... # gleiche, vorverarbeitete Eingabe wie fürs Training (float32)
o_f = run_interpreter(interpreter_f, x)
o_q = run_interpreter(interpreter_q, x)
print("Difference:", np.max(np.abs(o_f - o_q)))
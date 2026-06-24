from pathlib import Path

txt_path = Path("Arduino/quant_input_arduino.txt")
h_path = Path("ai_model/data/calibration_samples/offline_input.h")

values = []
inside = False

with txt_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line == "=== QUANTIZED INPUT DUMP BEGIN ===":
            inside = True
            continue
        if line == "=== QUANTIZED INPUT DUMP END ===":
            break

        if not inside:
            continue

        if line.startswith("shape:"):
            continue

        if line:
            try:
                values.append(int(line))
            except ValueError:
                pass

shape = (1, 1, 64, 41)

with h_path.open("w", encoding="utf-8") as f:
    f.write("#ifndef OFFLINE_INPUT_H\n")
    f.write("#define OFFLINE_INPUT_H\n\n")
    f.write("#include <Arduino.h>\n\n")
    f.write(f"const int OFFLINE_INPUT_SIZE = {len(values)};\n")
    f.write(f"const int OFFLINE_INPUT_SHAPE_0 = {shape[0]};\n")
    f.write(f"const int OFFLINE_INPUT_SHAPE_1 = {shape[1]};\n")
    f.write(f"const int OFFLINE_INPUT_SHAPE_2 = {shape[2]};\n")
    f.write(f"const int OFFLINE_INPUT_SHAPE_3 = {shape[3]};\n\n")
    f.write("const int8_t offline_input[OFFLINE_INPUT_SIZE] = {\n  ")

    for i, v in enumerate(values):
        f.write(str(v))
        if i < len(values) - 1:
            f.write(", ")
        if (i + 1) % 16 == 0 and i < len(values) - 1:
            f.write("\n  ")

    f.write("\n};\n\n")
    f.write("#endif // OFFLINE_INPUT_H\n")

print(f"Wrote {len(values)} values to {h_path}")
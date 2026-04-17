import torch
import numpy as np
from phase1.models.cnn import CNN

def get_multiplier_shift(scale: float):
    if scale == 0.0:
        return 0, 0

    shift = 0

    while scale < 0.5:
        scale *= 2.0
        shift -= 1

    while scale > 1.0:
        scale /= 2.0
        shift += 1

    multiplier = int(round(scale * (1 << 31)))

    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1

    return multiplier, shift

def main():
    # Load model
    model = CNN()
    model.load_state_dict(torch.load("phase1/best_cnn.pth", map_location="cpu"))
    model.eval()

    w = model.conv1.weight.data.numpy()   # [16,1,3,3]
    b = model.conv1.bias.data.numpy()

    # Input scale for uint8[0..255] mapped to int8[-128..127]
    input_scale = 1.0 / 127.0

    # Weight scale
    weight_scale = np.max(np.abs(w)) / 127.0

    # 👉 Define output scale (temporary constant for now)
    output_scale = 4.0 / 127.0   # keep simple for Day 19

    # 👉 Effective scale
    effective_scale = (input_scale * weight_scale) / output_scale

    w_q = np.round(w / weight_scale).astype(np.int8)

    # Correct bias quantization
    b_q = np.round(b / (input_scale * weight_scale)).astype(np.int32)

    # Reorder weights: [out, in, h, w] -> [out, h, w, in]
    w_q = np.transpose(w_q, (0, 2, 3, 1))

    w_flat = w_q.flatten()
    b_flat = b_q.flatten()

    print("int8_t conv1_weights[144] = {")
    print(",".join(map(str, w_flat)))
    print("};\n")

    print("int32_t conv1_bias[16] = {")
    print(",".join(map(str, b_flat)))
    print("};")

    multiplier, shift = get_multiplier_shift(effective_scale)

    print("int32_t multiplier[16] = {")
    print(",".join([str(multiplier)] * 16))
    print("};\n")

    print("int32_t shift[16] = {")
    print(",".join([str(shift)] * 16))
    print("};")



if __name__ == "__main__":
    main()
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


def quantize_conv_layer(weights, bias, input_scale, output_scale, layer_name):
    """
    weights: numpy array of shape [out_ch, in_ch, kH, kW]
    bias:    numpy array of shape [out_ch]
    """

    # Symmetric per-tensor weight quantization
    weight_scale = np.max(np.abs(weights)) / 127.0
    if weight_scale == 0:
        weight_scale = 1e-8

    weights_q = np.round(weights / weight_scale).astype(np.int8)

    # Bias quantization
    bias_scale = input_scale * weight_scale
    bias_q = np.round(bias / bias_scale).astype(np.int32)

    # Effective output scale
    effective_scale = bias_scale / output_scale

    multiplier, shift = get_multiplier_shift(effective_scale)

    # Reorder weights:
    # PyTorch: [out_ch, in_ch, kH, kW]
    # CMSIS:   [out_ch, kH, kW, in_ch]
    weights_q = np.transpose(weights_q, (0, 2, 3, 1))

    weights_flat = weights_q.flatten()
    bias_flat = bias_q.flatten()

    print(f"/* ===== {layer_name.upper()} ===== */")

    print(f"int8_t {layer_name}_weights[{len(weights_flat)}] = {{")
    print(",".join(map(str, weights_flat)))
    print("};\n")

    print(f"int32_t {layer_name}_bias[{len(bias_flat)}] = {{")
    print(",".join(map(str, bias_flat)))
    print("};\n")

    print(f"int32_t {layer_name}_multiplier[{len(bias_flat)}] = {{")
    print(",".join([str(multiplier)] * len(bias_flat)))
    print("};\n")

    print(f"int32_t {layer_name}_shift[{len(bias_flat)}] = {{")
    print(",".join([str(shift)] * len(bias_flat)))
    print("};\n")

    print(f"/* {layer_name} input_scale  = {input_scale} */")
    print(f"/* {layer_name} weight_scale = {weight_scale} */")
    print(f"/* {layer_name} output_scale = {output_scale} */")
    print(f"/* {layer_name} eff_scale    = {effective_scale} */")
    print(f"/* {layer_name} multiplier   = {multiplier} */")
    print(f"/* {layer_name} shift        = {shift} */")
    print()

def quantize_linear_layer(weights, bias, input_scale, output_scale, layer_name):
    """
    weights: numpy array of shape [out_features, in_features]
    bias:    numpy array of shape [out_features]
    """

    out_features, in_features = weights.shape

    weights_q_list = []
    bias_q_list = []
    multipliers = []
    shifts = []
    weight_scales = []

    for o in range(out_features):
        w_o = weights[o]   # shape: [in_features]

        # Per-output-channel weight scale
        w_scale_o = np.max(np.abs(w_o)) / 127.0
        if w_scale_o == 0:
            w_scale_o = 1e-8

        weight_scales.append(w_scale_o)

        # Quantize this output neuron weights
        w_q_o = np.round(w_o / w_scale_o).astype(np.int8)
        weights_q_list.append(w_q_o)

        # Quantize bias for this output neuron
        bias_scale_o = input_scale * w_scale_o
        b_q_o = int(np.round(bias[o] / bias_scale_o))
        bias_q_list.append(b_q_o)

        # Per-output-channel requantization params
        effective_scale_o = bias_scale_o / output_scale
        m_o, s_o = get_multiplier_shift(effective_scale_o)
        multipliers.append(m_o)
        shifts.append(s_o)

    weights_q = np.stack(weights_q_list, axis=0)   # [out_features, in_features]
    weights_flat = weights_q.flatten()
    bias_flat = np.array(bias_q_list, dtype=np.int32)

    print(f"/* ===== {layer_name.upper()} ===== */")

    print(f"int8_t {layer_name}_weights[{len(weights_flat)}] = {{")
    print(",".join(map(str, weights_flat)))
    print("};\n")

    print(f"int32_t {layer_name}_bias[{len(bias_flat)}] = {{")
    print(",".join(map(str, bias_flat)))
    print("};\n")

    print(f"int32_t {layer_name}_multiplier[{len(multipliers)}] = {{")
    print(",".join(map(str, multipliers)))
    print("};\n")

    print(f"int32_t {layer_name}_shift[{len(shifts)}] = {{")
    print(",".join(map(str, shifts)))
    print("};\n")

    print(f"/* {layer_name} input_scale  = {input_scale} */")
    print(f"/* {layer_name} output_scale = {output_scale} */")
    print(f"/* {layer_name} weight_scale min = {np.min(weight_scales)} */")
    print(f"/* {layer_name} weight_scale max = {np.max(weight_scales)} */")
    print()

def main():
    # Load trained model
    model = CNN()
    model.load_state_dict(torch.load("phase1/best_cnn.pth", map_location="cpu"))
    model.eval()

    # -----------------------------
    # Conv1 scales
    # -----------------------------
    # Your STM32 input preprocessing:
    # input_data[i] = (input_u8[i] * 127) / 255
    # so real input is approximately q * (1/127)
    conv1_input_scale = 1.0 / 127.0

    # Day 19 tuned output scale for Conv1
    conv1_output_scale = 4.0 / 127.0

    conv1_w = model.conv1.weight.detach().cpu().numpy()
    conv1_b = model.conv1.bias.detach().cpu().numpy()

    quantize_conv_layer(
        weights=conv1_w,
        bias=conv1_b,
        input_scale=conv1_input_scale,
        output_scale=conv1_output_scale,
        layer_name="conv1"
    )

    # -----------------------------
    # Conv2 scales
    # -----------------------------
    # Conv2 input comes from pooled Conv1 output.
    # ReLU does not change scale.
    # MaxPool does not change scale.
    # So Conv2 input scale = Conv1 output scale.
    conv2_input_scale = conv1_output_scale

    # Temporary output scale for Conv2.
    # Start with same style as Conv1 and tune later if needed.
    conv2_output_scale = 6.0 / 127.0

    conv2_w = model.conv2.weight.detach().cpu().numpy()
    conv2_b = model.conv2.bias.detach().cpu().numpy()

    quantize_conv_layer(
        weights=conv2_w,
        bias=conv2_b,
        input_scale=conv2_input_scale,
        output_scale=conv2_output_scale,
        layer_name="conv2"
    )

        # -----------------------------
    # FC1 scales
    # -----------------------------
    # Conv2 pool output is the input to FC1
    fc1_input_scale = conv2_output_scale
    fc1_output_scale = 0.10737494971808487

    fc1_w = model.fc1.weight.detach().cpu().numpy()
    fc1_b = model.fc1.bias.detach().cpu().numpy()

    quantize_linear_layer(
        weights=fc1_w,
        bias=fc1_b,
        input_scale=fc1_input_scale,
        output_scale=fc1_output_scale,
        layer_name="fc1"
    )

    # -----------------------------
    # FC2 scales
    # -----------------------------
    fc2_input_scale = fc1_output_scale
    fc2_output_scale = 0.14029029786117433

    fc2_w = model.fc2.weight.detach().cpu().numpy()
    fc2_b = model.fc2.bias.detach().cpu().numpy()

    quantize_linear_layer(
        weights=fc2_w,
        bias=fc2_b,
        input_scale=fc2_input_scale,
        output_scale=fc2_output_scale,
        layer_name="fc2"
    )


if __name__ == "__main__":
    main()
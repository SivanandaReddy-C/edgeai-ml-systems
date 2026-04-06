import json
import os
import numpy as np
import onnxruntime as ort

# Load config
with open("../config/config.json") as f:
    config = json.load(f)

# Load model
model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    config["model_path"]
)
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Dummy input
input_data = np.random.randn(*config["input_shape"]).astype(np.float32)

# Run Inference
output = session.run(None,{input_name: input_data})

print("Inference successful")
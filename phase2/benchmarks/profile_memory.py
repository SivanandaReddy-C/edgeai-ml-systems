import onnxruntime as ort
import numpy as np
from memory_profiler import profile

@profile
def run_inference():
    session = ort.InferenceSession(
        "phase2/models/transformer.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    input_data = np.random.randn(1, 28, 28).astype(np.float32)

    for _ in range(100):
        session.run(None, {input_name: input_data})

if __name__=="__main__":
    run_inference()

    
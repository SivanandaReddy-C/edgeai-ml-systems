import time
import onnxruntime as ort

def measure_load_time(model_path):
    start = time.time()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    end = time.time()

    return (end - start) *1000 #ms

models = {
 "CNN FP32": "phase2/models/cnn.onnx",
    "CNN INT8": "phase2/models/cnn_int8.onnx",
    "Transformer FP32": "phase2/models/transformer.onnx",
    "Transformer INT8": "phase2/models/transformer_int8.onnx",  
}

print("\nModel Loadint Time (Cold Start):\n")

for name, path in models.items():
    times = []
    
    try:
        for _ in range(5):
            t = measure_load_time(path)
            times.append(t)
        
        avg_time = sum(times) / len(times)
        print(f"{name:20s}: {avg_time:.2f} ms")
    
    except Exception as e:
        print(f"{name:20s}: FAILED ({str(e).split(':')[0]})")
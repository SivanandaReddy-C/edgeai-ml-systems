import time
import torch

from models.cnn import CNN
from utils.dataset import get_dataloaders
from training.train import train

def benchmark_training():
    train_loader,_=get_dataloaders(batch_size=32)

    model=CNN()

    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    start_time=time.perf_counter()
    train(model,train_loader,optimizer,criterion)
    end_time=time.perf_counter()

    training_time=end_time-start_time
    print(f"Training time for 1 epoch:{training_time:.2f} seconds")

def benchmark_single_inference():
    model=CNN()
    model.eval()

    dummy_input=torch.randn(1,1,28,28)

    start_time=time.perf_counter()
    with torch.no_grad():
        output=model(dummy_input)
    end_time=time.perf_counter()

    latency=(end_time-start_time)*100

    print(f"Single image inference latency:{latency:.3f}ms")

def benchmark_batch_inference():
    model=CNN()
    model.eval()

    dummy_batch=torch.randn(32,1,28,28)

    start_time=time.perf_counter()
    with torch.no_grad():
        output=model(dummy_batch)
    end_time=time.perf_counter()

    latency=(end_time-start_time)*1000

    print(f"Batch (32) inference latency: {latency:.3f}ms")

if __name__=="__main__":
    print("Running Benchmarks...")
    print("-"*40)

    benchmark_training()
    benchmark_single_inference()
    benchmark_batch_inference()



import time
import torch

from phase1.models.cnn import CNN
from phase1.models.transformer import TransformerClassifier
from phase1.utils.dataset import get_dataloaders
from phase1.training.train import train

BATCH_SIZE=32
RUNS=100

def benchmark_training(model, model_name):
    """
    Measures training time for one epoch.

    This function initializes the dataloader, loss function, and optimizer, 
    and runs one epoch of traning using the provided model.

    Args:
        model (torch.nn.Module): The model to be trained (CNN or Transformer).
        model_name (str): Name of the model for logging purposes.

    Prints:
        Training time for one epoch in seconds
    """
    train_loader,_ = get_dataloaders(batch_size=BATCH_SIZE)

    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.perf_counter()
    train(model, train_loader, optimizer, criterion)
    end_time = time.perf_counter()

    training_time = end_time - start_time
    print(f"{model_name} training time for 1 epoch:{training_time:.2f} seconds")

def benchmark_single_inference():
    """
    Measures average latency for single-sample inference.

    Runs multiple forward passes on both CNN and Transformer models
    using a single input sample to obtain stable latency measurements.

    Prints:
        Average inference latency (in milliseconds) for:
            -CNN
            -Transformer
    """
    runs = RUNS

    cnn = CNN()
    transformer = TransformerClassifier()

    cnn.eval()
    transformer.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    #CNN
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            cnn(dummy_input)
    end_time = time.perf_counter()
    cnn_latency = (end_time - start_time) * 1000 / runs

    #Transformer
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            transformer(dummy_input)
    end = time.perf_counter()
    transformer_latency = (end - start) * 1000 / runs

    print(f"CNN single image inference latency:{cnn_latency:.3f}ms")
    print(f"Transformer single image inference latency:{transformer_latency:.3f}ms")


def benchmark_batch_inference():
    """
    Measures average latency for batch inference.

    Runs multiple forward passes on both CNN and Transformer models
    using a batch of inputs (size=32) to evaluate througput performance.

    Prints:
        Average batch inference latency (in milliseconds) for:
            - CNN
            - Transformer
    """
    runs = 100

    cnn = CNN()
    transformer = TransformerClassifier()

    cnn.eval()
    transformer.eval()

    dummy_batch = torch.randn(32, 1, 28, 28)

    #CNN
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            cnn(dummy_batch)
    end_time = time.perf_counter()
    cnn_latency = (end_time - start_time) * 1000 / runs

    #Transformer
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            transformer(dummy_batch)
    end = time.perf_counter()
    transformer_latency = (end - start) * 1000 / runs

    print(f"CNN batch inference latency: {cnn_latency:.3f}ms")
    print(f"Transformer batch inference latency: {transformer_latency:.3f}ms")

def count_parameters(model):
    """
    Counts total number of trainable parameters in a model.
        Args: 
            model (torch.nn.Module): Model instance
        
        Returns:
            int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())

def benchmark_parameters():
    """
    Computes and compares the number of trainable parameters.

    Instantiates both CNN and Transformer models and calculates
    the total number of parameters in each.

    Prints:
        Total parameter count for:
            -CNN
            -Transformer
    """
    cnn = CNN()
    transformer = TransformerClassifier()

    cnn_params = count_parameters(cnn)
    transformer_params = count_parameters(transformer)

    print(f"CNN parameters:{cnn_params}")
    print(f"Transformer parameters:{transformer_params}")

if __name__=="__main__":
    print("Running benchmarks...")
    print("-"*40)

    # TRAINING
    benchmark_training(CNN(), "CNN")
    benchmark_training(TransformerClassifier(), "Transformer")
    print("-"*40)

    # INFERENCE
    benchmark_single_inference()
    print("-"*40)

    benchmark_batch_inference()
    print("-"*40)

    # PARAMETERS
    benchmark_parameters()
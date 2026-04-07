from memory_profiler import profile
import cProfile
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time

from phase1.models.transformer import TransformerClassifier
from phase1.models.cnn import CNN
from phase1.utils.dataset import get_dataloaders

def parse_args():
    """
    Parses command-line arguments for training configuration

    Returns:
        argparse.Namespace: Parsed arguments including batch size,
        number of epochs, learning rate, and model name.
    """
    parser = argparse.ArgumentParser(description="CNN Training Script")

    parser.add_argument("--batch_size",type=int,help="Batch size for training")
    parser.add_argument("--epochs",type=int,help="Number of training epochs")
    parser.add_argument("--lr",type=float,help="Learning rate")
    parser.add_argument("--model_name", type=str, help="Model to use: cnn or transformer")

    return parser.parse_args()

def load_config(config_path):
    """
    Loads YAML configuration file.

    Args
        config_path (str): Path to YAML config file.

    Returns:
        dict: Configuration dictionary containing training and model settings.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def override_config(config, args):
    """
    Overrides YAML configuration with command-line arguments.

    Args:
        config (dict): Original configuration dictionary.
        args (argparse.Namespace): Parsed CLI arguments

    Returns:
        dict: Updated configuration dictionary.
    """
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    if args.lr:
        config["training"]["learning_rate"] = args.lr

    if args.model_name:
        config["model"]["name"] = args.model_name
    
    return config

@profile
def train(model,train_loader, optimizer,criterion):
    """
    Runs one epoch of training.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (torch.nn.Module): Loss function.

    Returns:
        tuple:
            avg_loss (float): Average training loss.
            accuracy (float): Training accuracy (%).
    """
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() 

    accuracy = 100 * correct / total

    return total_loss / len(train_loader), accuracy


def evaluate(model,test_loader, criterion):
    """
    Evaluates the model on validation/test dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (DataLoader): Validation/test data loader.
        criterion (torch.nn.Module): Loss function.

    Return:
        tuple:
            avg_loss (float): Average validation loss.
            accuracy (float): Validation accuracy (%).
    """
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images,labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    
    accuracy = 100 * correct / total

    return total_loss / len(test_loader),accuracy

def main():
    """
    Main training pipeline.

    -Loads configuration
    -Initializes model, optimizer, and data loaders
    -Runs training and validation loops
    -Tracks best model based on validation accuracy
    -Measures training and validation time per epoch
    """
    args = parse_args()
    config = load_config("configs/cnn.yaml")  
    config = override_config(config,args)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    model_name = config["model"]["name"]

    if model_name == "cnn":
        model = CNN()
    elif model_name == "transformer":
        model = TransformerClassifier()
    else:
        raise ValueError("Invalid model name in config")
    
    train_loader,test_loader = get_dataloaders(batch_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=lr)

    best_accuracy = 0

    for epoch in range(epochs):

        # Training time
        train_start = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer,criterion)
        train_end = time.time()

        #Validation time
        val_start = time.time()
        val_loss,val_acc = evaluate(model, test_loader, criterion)
        val_end = time.time()

        train_time = train_end - train_start
        val_time = val_end - val_start

        print(f"Using model: {model_name}")
        print(f"Train time:{train_time:.2f} seconds")
        print(f"Validation time:{val_time:.2f} seconds")
        print(f"Total epoch time:{train_time+val_time:.2f} seconds")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss:{train_loss:.4f}")
        print(f"Val loss:{val_loss:.4f}")
        print(f"Val Accuracy:{val_acc:.2f}%")

        if val_acc > best_accuracy:
            
            best_accuracy = val_acc
            
            torch.save(model.state_dict(),"best_transformer.pth")
            
            print("Best model saved!")

        print("-" * 40)


if __name__=="__main__":
    profiler = cProfile.Profile()

    profiler.enable()

    main()

    profiler.disable()

    profiler.print_stats(sort="time")
    profiler.dump_stats("training_profile.prof")




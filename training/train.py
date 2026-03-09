import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from utils.dataset import get_dataloaders

import yaml

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(model,train_loader,optimizer,criterion):
    model.train()

    total_loss=0
    correct=0
    total=0

    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        _, predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item() 

    accuracy=100*correct/total

    return total_loss/len(train_loader), accuracy



def evaluate(model,test_loader,criterion):
    model.eval()

    total_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in test_loader:
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_loss+=loss.item()
            _,predicted = torch.max(outputs,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    
    accuracy=100*correct/total

    return total_loss/len(test_loader),accuracy


if __name__=="__main__":
    config=load_config("configs/cnn.yaml")
    batch_size=config["training"]["batch_size"]
    epochs=config["training"]['epochs']
    lr=config["training"]["learning_rate"]

    train_loader,test_loader=get_dataloaders(batch_size)

    model=CNN()

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=lr)

    best_accuracy=0
    for epoch in range(epochs):
        train_loss, train_acc=train(model,train_loader,optimizer,criterion)

        val_loss,val_acc=evaluate(model,test_loader,criterion)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss:{train_loss:.4f}")
        print(f"Val Loss:{val_loss:.4f}")
        print(f"Val Accuracy:{val_acc:.2f}%")

        if val_acc>best_accuracy:
            
            best_accuracy=val_acc
            
            torch.save(model.state_dict(),"best_model.pth")
            
            print("Best model saved!")

        print("-" * 40)




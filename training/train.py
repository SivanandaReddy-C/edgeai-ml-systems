import os
print("Working Directory:", os.getcwd())


import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from utils.dataset import get_dataloaders

def train(model,train_loader,optimizer,criterion):
    model.train()

    total_loss=0

    for images,labels in train_loader:

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    return total_loss/len(train_loader)

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
    batch_size=32
    epochs=5
    lr=0.001

    train_loader,test_loader=get_dataloaders(batch_size)

    model=CNN()

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        train_loss=train(model,train_loader,optimizer,criterion)

        val_loss,val_acc=evaluate(model,test_loader,criterion)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss:{train_loss:.4f}")
        print(f"Val Loss:{val_loss:.4f}")
        print(f"Val Accuracy:{val_acc:.2f}%")
        print("-" * 40)




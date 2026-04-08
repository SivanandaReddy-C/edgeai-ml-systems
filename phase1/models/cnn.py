import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    Architecture:
    Conv → ReLU → Pool → Conv → ReLU → Pool → FC → ReLU → FC
    """
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32*7*7,128)
        self.fc2 = nn.Linear(128,10)

        self.flatten = nn.Flatten()
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)

        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=="__main__":
    model = CNN()

    x = torch.randn(1, 1, 28, 28)

    y = model(x)

    print("Output shape:", y.shape)

    print("Total parameters:", count_parameters(model))
  

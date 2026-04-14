import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
BATCH_SIZE=32
class MNISTDataset(Dataset):
    """
    custom wrapper for MNIST dataset
    
    Handles:
    - data download
    - transformations
    - dataset abstraction layer
    """
    def __init__(self,train=True):
        """
        Initializes MNIST dataset.

        Args:
            train (bool): If True, loads training dataset;
                            otherwise loads test dataset.
        """
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = datasets.MNIST(root="./data", train=train, download=True,
                                    transform=self.transform)
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Retrieves a single data sample.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple:
                image (Tensor): Input image.
                label (int): Corresponding label.
        """
        image,label = self.dataset[index]
        return image,label
    
def get_dataloaders(batch_size=BATCH_SIZE):
    """
    Creates training and test DataLoaders for MNIST.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        tuple:
            train_loader (DataLoader)
            test_loader (DataLoader)
    """
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader



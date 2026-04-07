
from phase1.utils.dataset import get_dataloaders

train_loader,test_loader= get_dataloaders(batch_size=32)

for images,labels in train_loader:
    print("Images shape:",images.shape)
    print("Labels shape:",labels.shape)
    print("Total training samples:", len(train_loader.dataset))
    break


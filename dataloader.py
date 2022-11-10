from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

BATCH_SIZE = 128

def get_dataloaders():
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    # Datasets
    train_dataset = MNIST(root='./MNIST/', train=True, transform=train_transform, download=True)
    test_dataset = MNIST(root='./MNIST/', train=False, transform=test_transform, download=True)

    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader
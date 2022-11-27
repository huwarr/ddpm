from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

BATCH_SIZE = 128

def get_dataloaders(dataset_name):
    assert dataset_name == 'CIFAR10' or dataset_name == 'MNIST', "This dataset is unavaliable. Please, choose MNIST or CIFAR10"

    if dataset_name == 'MNIST':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

        train_dataset = MNIST(root='./MNIST/', train=True, transform=train_transform, download=True)
        test_dataset = MNIST(root='./MNIST/', train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, test_loader
    else:
        # CIFAR10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32))
        ])

        train_dataset = CIFAR10(root='./CIFAR10/', train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root='./CIFAR10/', train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, test_loader
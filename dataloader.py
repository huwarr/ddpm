from torchvision.datasets import MNIST
from torchvision import transforms

BATCH_SIZE = 128

def get_dataloaders():
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        # DDPM, Appendix B -> "We used random horizontal flips during training"
        transforms.RandomHorizontalFlip()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    # Datasets
    train_dataset = MNIST(root='./MNIST/', train=True, transform=train_transform, download=True)
    test_dataset = MNIST(root='./MNIST/', train=False, transform=transforms.ToTensor(), download=True)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
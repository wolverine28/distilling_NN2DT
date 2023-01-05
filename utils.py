import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# get MNIST dataset
def get_mnist_dataset(batch_size=64, shuffle=True, num_workers=0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader, test_loader
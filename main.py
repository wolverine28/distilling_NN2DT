import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from anytree import PreOrderIter,find_by_attr
from einops import rearrange
from tqdm import tqdm

from tree import softTree


# get MNIST dataset
def get_mnist_dataset(batch_size=64, shuffle=True, num_workers=0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    # set hyperparameters
    batch_size = 64
    epochs = 20
    lr = 1e-2
    momentum = 0.9
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataset(batch_size=batch_size, shuffle=True, num_workers=0)

    # set model
    model = softTree(depth = 5, feature_size = 784, n_classes = 10, batch_size = batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        tq = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(tq):
            data, target = data.to(device), target.to(device)
            data = rearrange(data, 'b c h w -> b (c h w)')

            optimizer.zero_grad()

            # output_distribution = model.forward(data)
            # leaf_prob = model.forward_prob(data)
            # onehot_target = F.one_hot(target, num_classes=10).float()

            model.forward_prob(data, model.tree.root)
            
            loss = model.cal_loss(target)
            loss += model.regularizer(data)*1
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = model.predict().argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            tq.set_postfix_str(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100.*corr/pred.shape[0]:7.2f}%')

        train_loss /= len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: ({train_acc:.0f}%)')


import torch
import torch.nn as nn
import torch.optim as optim
from convnet import LeNet5
from einops import rearrange
from tqdm import tqdm

from utils import get_mnist_dataset

if __name__ == '__main__':
    # set hyperparameters
    batch_size = 64
    epochs = 5
    lr = 3e-4
    momentum = 0.9
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataset(batch_size=batch_size, shuffle=True, num_workers=0)

    # set model
    model = LeNet5(10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        tq = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(tq):
            model.train()
            data, target = data.to(device), target.to(device)
            # data = rearrange(data, 'b c h w -> b (c h w)')

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            tq.set_postfix_str(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100.*corr/pred.shape[0]:7.2f}%')

        train_loss /= len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: ({train_acc:.3f}%)')

        test_loss = 0
        correct = 0
        tq = test_loader
        for batch_idx, (data, target) in enumerate(tq):
            model.eval()
            data, target = data.to(device), target.to(device)
            # data = rearrange(data, 'b c h w -> b (c h w)')


            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            # tq.set_postfix_str(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100.*corr/pred.shape[0]:7.2f}%')

        test_loss /= len(test_loader)
        test_Acc = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: ({test_Acc:.3f}%)')

    torch.save(model.state_dict(), 'lenet5.pth')
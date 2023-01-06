import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from tqdm import tqdm

from convnet import LeNet5
from tree import softTree
from utils import get_mnist_dataset

if __name__ == '__main__':
    # set hyperparameters
    batch_size           = 64
    epochs               = 50
    lr                   = 0.02
    momentum             = 0.9
    temperature          = 2
    # distill              = True
    UseSoftTarget        = True
    regularizer_strength = 1.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataset(batch_size=batch_size, shuffle=True, num_workers=0)

    # set model
    model = softTree(depth = 4, feature_size = 784, n_classes = 10, batch_size = batch_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    convnet = LeNet5(10).to(device)
    convnet.load_state_dict(torch.load('lenet5.pth'))
    convnet.eval()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        tq = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(tq):
            model.train()
            data, target = data.to(device), target.to(device)

            if UseSoftTarget:
                soft_target = torch.softmax(convnet(data)/temperature, dim=1)
            else:
                soft_target = F.one_hot(target, num_classes=10).float()
            data = rearrange(data, 'b c h w -> b (c h w)')

            optimizer.zero_grad()
            
            model.init_prob(data.shape[0])
            model.forward_prob(data, model.tree.root)
            pred = model.predict_soft()

            # student loss
            loss = (-soft_target*torch.log(pred)).sum(1).mean()
            loss += model.regularizer(data)*regularizer_strength

            # if distill:
            #     #distillation loss between pred and soft_target
            #     kl_div = F.kl_div(torch.log(pred), soft_target, reduction='batchmean')
            #     # total loss
            #     loss += kl_div

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred = model.predict_hard().argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            tq.set_postfix_str(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100.*corr/pred.shape[0]:7.2f}%')

        train_loss /= len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: ({train_acc:.3f}%)')

        test_loss = 0
        correct = 0
        tq = tqdm(test_loader)
        for batch_idx, (data, target) in enumerate(tq):
            model.eval()
            data, target = data.to(device), target.to(device)
            data = rearrange(data, 'b c h w -> b (c h w)')

            # output_distribution = model.forward(data)
            # leaf_prob = model.forward_prob(data)
            onehot_target = F.one_hot(target, num_classes=10).float()
            model.init_prob(data.shape[0])
            model.forward_prob(data, model.tree.root)
            pred = model.predict_soft()

            loss = (-onehot_target*torch.log(pred)).sum(1).mean()
            # loss += model.regularizer(data)*1
            
            test_loss += loss.item()

            pred = model.predict_hard().argmax(dim=1, keepdim=True)
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr

            # tq.set_postfix_str(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100.*corr/pred.shape[0]:7.2f}%')

        test_loss /= len(test_loader)
        test_Acc = 100. * correct / len(test_loader.dataset)
        print(f'                    Test set: Average loss: {test_loss:.4f}, Accuracy: ({test_Acc:.3f}%)')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'softTree.pth')
            print('Model saved')
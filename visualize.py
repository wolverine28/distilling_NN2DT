import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from tqdm import tqdm

from convnet import LeNet5
from tree import softTree
from utils import get_mnist_dataset
from anytree import AnyNode, PreOrderIter, RenderTree
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # set hyperparameters
    batch_size           = 64
    epochs               = 50
    lr                   = 3e-3
    momentum             = 0.9
    temperature          = 2
    # distill              = True
    UseSoftTarget        = True
    regularizer_strength = 1.

    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': 'gray'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataset(batch_size=batch_size, shuffle=True, num_workers=0)

    # set model
    model = softTree(depth = 4, feature_size = 784, n_classes = 10, batch_size = batch_size).to(device)
    model.load_state_dict(torch.load('softTree.pth'))


    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    data = rearrange(data, 'b c h w -> b (c h w)')

    model.init_prob(data.shape[0])
    model.forward_prob(data, model.tree.root)
    pred = model.predict_hard()

    all_nodes = [node for node in PreOrderIter(model.tree.root)]
    for node in all_nodes:
        if node.is_leaf:
            filter = torch.softmax(model.leafs[node.id].leaves,dim=0).detach().cpu().numpy()

            plt.plot(filter)
            plt.xticks(np.arange(10))
            plt.savefig(f'figures/{node.id}.png')
            plt.close()
        else:
            filter = rearrange(model.stumps[node.id].filter,'(h w) c -> (c h) w', h=28, w=28)
            filter = ((filter-filter.min())/(filter.max()-filter.min())).detach().cpu().numpy()

            plt.imshow(filter)
            plt.savefig(f'figures/{node.id}.png')
            plt.close()
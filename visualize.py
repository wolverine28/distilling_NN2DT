import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from anytree import PreOrderIter
from einops import rearrange
from tqdm import tqdm

from models.tree import softTree
from parse_config import ConfigParser
from utils import get_mnist_dataset


def main(config):
    #  # set hyperparameters
    # batch_size           = 64
    # UseCorr              = False
    # regularizer_strength = 1.
    # # imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': 'gray'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataset(batch_size=config['batch_size'], shuffle=True, num_workers=0)

    # set model
    model = softTree(depth = 6, feature_size = 784, n_classes = 10, batch_size = config['batch_size']).to(device)
    model.load_state_dict(torch.load('softTree.pth'))


    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    data = rearrange(data, 'b c h w -> b (c h w)')
    data = data[[0],:]

    model.init_prob(data.shape[0])
    model.forward_prob(data, model.tree.root)
    pred = model.predict_hard()

    ddd = data
    ddd_plot = rearrange(ddd.view(-1),'(h w) -> h w', h=28, w=28)
    ddd_plot = ((ddd_plot-ddd_plot.min())/(ddd_plot.max()-ddd_plot.min())).detach().cpu().numpy()
    fig = plt.figure(figsize=(10,10))
    plt.matshow(ddd_plot)
    fig.tight_layout()
    plt.savefig(f'figures/nodes/data.png')
    plt.close()

    prob_dict = {}
    all_nodes = [node for node in PreOrderIter(model.tree.root)]
    for node in all_nodes:
        if node.is_leaf:
            filter = torch.softmax(model.leafs[node.id].leaves,dim=0).detach().cpu().numpy()

            fig = plt.figure(figsize=(10,6))
            plt.bar(np.arange(10),filter)
            plt.xticks(np.arange(10))
            fig.tight_layout()
            plt.savefig(f'figures/nodes/{node.id}.png')
            plt.close()
        else:
            filter = model.stumps[node.id].filter.view(-1)
            bias = model.stumps[node.id].bias.view(-1)
            if config['use_corr']:
                filter = ddd.view(-1)*filter
                rightprob = torch.sigmoid(model.beta*((ddd*filter).sum()+bias))
                leftprob = 1-rightprob
                prob_dict[node.id] = (leftprob.item(),rightprob.item())
            filter = rearrange(filter,'(h w) -> h w', h=28, w=28)
            filter = ((filter-filter.min())/(filter.max()-filter.min())).detach().cpu().numpy()

            fig = plt.figure(figsize=(10,10))
            plt.matshow(filter)
            plt.clim(-3,3)
            fig.tight_layout()
            plt.savefig(f'figures/nodes/{node.id}.png')
            plt.close()

    with open(config.save_dir / 'prob_dict.pickle', 'wb') as handle:
        pickle.dump(prob_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Distilling a Neural Network Into a Soft Decision Tree')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, '',run_id='')


    main(config)

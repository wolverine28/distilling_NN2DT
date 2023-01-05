import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anytree import AnyNode, RenderTree, find_by_attr, PreOrderIter
from anytree.exporter import DotExporter


class stump(nn.Module):
    def __init__(self, feature_size):
        super(stump, self).__init__()
        self.feature_size = feature_size
        self.filter = nn.Parameter(torch.randn(size=(self.feature_size, 1)).cuda())
        self.bias = nn.Parameter(torch.randn(1).cuda())
        self.sigmoid = nn.Sigmoid()

        # x dim : [batch_size, feature_size]
    def forward(self, x):
        return self.sigmoid(torch.einsum("bf, fj -> bj",x, self.filter) + self.bias)

class leaf(nn.Module):
    def __init__(self, n_classes):
        super(leaf, self).__init__()
        self.leaves = nn.Parameter(torch.randn(size=(n_classes,)).cuda())

    def forward(self):
        return self.leaves
        # return F.softmax(self.leaves, dim=-1)

class EMA:
    def __init__(self, mu):
        self.mu = mu
        
    def update(self, x, last_average):
        new_average = self.mu*x + (1-self.mu)*last_average
        return new_average

class Mytree:
    def __init__(self, depth, feature_size, n_classes):
        self.depth = depth
        self.feature_size = feature_size
        self.n_classes = n_classes

    def grow(self):
        self.root = AnyNode(id='0',depth=0, ema = EMA(0.5+0.05*2**0), prob = torch.tensor(0.5))
        for d in range(self.depth):
            for node in self.root.leaves:
                if node.depth == d:
                    if d == self.depth-1:
                        AnyNode(id=node.id+'0', parent=node, depth=d+1, ema = EMA(0.5+0.05*2**d), prob = torch.tensor(0.5))
                        AnyNode(id=node.id+'1', parent=node, depth=d+1, ema = EMA(0.5+0.05*2**d), prob = torch.tensor(0.5))
                    else:
                        AnyNode(id=node.id+'0', parent=node, depth=d+1, ema = EMA(0.5+0.05*2**d), prob = torch.tensor(0.5))
                        AnyNode(id=node.id+'1', parent=node, depth=d+1, ema = EMA(0.5+0.05*2**d), prob = torch.tensor(0.5))            

    def display(self):
        print(RenderTree(self.root))


class softTree(nn.Module):
    def __init__(self, depth=4, n_classes=10, feature_size=784, batch_size=32):
        super(softTree, self).__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.feature_size = feature_size

        self.tree = Mytree(depth = self.depth, feature_size = self.feature_size , n_classes = self.n_classes)
        self.tree.grow()

        self.stumps = nn.ModuleDict()
        self.leafs = nn.ModuleDict()

        innerNode = [node for node in PreOrderIter(self.tree.root, filter_=lambda node: not node.is_leaf)]
        for node in innerNode:
            self.stumps[node.id] = stump(self.feature_size)
            # node.net = self.stumps[node.id]
        
        ieafNode = [node for node in PreOrderIter(self.tree.root, filter_=lambda node: node.is_leaf)]
        for node in ieafNode:
            self.leafs[node.id] = leaf(self.n_classes)
            # node.net = self.leafs[node.id]

        self.all_node_prob = {'0':torch.ones(batch_size).cuda()}

        self.criterion = nn.NLLLoss()
    # def semi_forward(self, x, nets):
    #     return torch.concat([self.stumps[id](x[[i]]) for i,id in enumerate(nets)])

    # def leaf_forward(self, nets):
    #     return torch.concat([self.leafs[id]() for id in nets])


    # def forward(self, x):
    #     leafprob, id = self.forward_prob(x)
        
    #     leafprob
    #     torch.stack([self.leafs[i]() for i in id])



    #     return prob

    def cal_loss(self, target):
        leaf_id = [node.id for node in PreOrderIter(self.tree.root, filter_=lambda node: node.is_leaf)]

        output_distribution = torch.stack([self.leafs[i]() for i in leaf_id])
        leafprob = torch.stack([self.all_node_prob[i] for i in leaf_id]).T

        log_prob = F.log_softmax(output_distribution, dim=-1)
        pred = torch.einsum('bl,lc->bc',leafprob,log_prob)

        loss = self.criterion(pred, target)

        return loss
    
    def predict(self):
        # leafprob, id = self.forward_prob(x)
        leaf_id = [node.id for node in PreOrderIter(self.tree.root, filter_=lambda node: node.is_leaf)]
        output_distribution = torch.stack([self.leafs[i]() for i in leaf_id])
        leafprob = torch.stack([self.all_node_prob[i] for i in leaf_id]).T

        return output_distribution[leafprob.max(1).indices,:]

    
    def forward_prob(self, x, node):
        if node.is_leaf:
            return 1

        prob = self.stumps[node.id](x).view(-1)

        self.all_node_prob[node.children[0].id] = self.all_node_prob[node.id]*(1-prob)
        self.all_node_prob[node.children[1].id] = self.all_node_prob[node.id]*prob

        self.forward_prob(x, node.children[0])
        self.forward_prob(x, node.children[1])

        return None

    def regularizer(self, x):
        noleaf = [node for node in PreOrderIter(self.tree.root, filter_=lambda node: not node.is_leaf)]
        _loss = 0
        for node in noleaf:
            path_p = self.all_node_prob[node.id].cuda()+1e-5
            p = self.stumps[node.id](x).view(-1)
            alpha = (path_p*p).sum()/path_p.sum()
            node.prob = node.ema.update(alpha, node.prob.detach())

            _loss += -(0.5*torch.log(node.prob+1e-6) + 0.5*torch.log(1-node.prob+1e-6))*(0.5**node.depth)
        return _loss/len(noleaf)

    # def pathprob(self, node, x):
    #     move = node.id[1:]
    #     if len(move) == 0:
    #         return torch.ones(size=(x.shape[0],))
    #     prob = torch.stack([self.stumps[n.id](x).view(-1) for n in node.ancestors])

    #     right_prob = prob
    #     left_prob = 1-prob

    #     for i,m in enumerate(move):
    #         if m == '1':
    #             prob[i][prob[i]<0.5] = 1-prob[i][prob[i]<0.5]
    #         else:
    #             prob[i][prob[i]>=0.5] = 1-prob[i][prob[i]>=0.5]

    #     return prob.prod(0)

if __name__ == '__main__':
    from torchviz import make_dot
    from torch.autograd import Variable

    net = softTree(depth = 4, feature_size = 784, n_classes = 10, batch_size = 32)

    x = torch.randn(size=(32,784)).cuda()
    target = torch.randint(0,10,size=(32,)).cuda()
    onehot_target = F.one_hot(target, num_classes=10).float()

    net.forward_prob(x, net.tree.root)
    loss = net.cal_loss(onehot_target)

    make_dot(loss, params=dict(net.named_parameters())).render("graph", format="png")
    print(net.all_node_prob)
    # print(find_by_attr(test_tree.root, name='id', value='0').net(x))
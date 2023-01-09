import argparse
import os
import pickle

import graphviz
from anytree import AnyNode, PreOrderIter, RenderTree

from models.tree import softTree
from parse_config import ConfigParser


def main(config):
    model = softTree(depth = config['depth'], feature_size = config['feature_size'], n_classes = config['n_class'], batch_size = 0)

    with open(config.save_dir / 'prob_dict.pickle', 'rb') as handle:
        prob_dict = pickle.load(handle)

    def forward_build(node):
        if node.is_leaf:    return 
        
        digraph1.node(node.children[0].id   , shape = 'box'
                                            , image = f"figures/nodes/{node.children[0].id}.png"
                                            , label = ''
                                            , width = '10px', height='10px',imagescale='true')
        digraph1.node(node.children[1].id   , shape = 'box'
                                            , image = f"figures/nodes/{node.children[1].id}.png"
                                            , label = ''
                                            , width = '10px', height='10px',imagescale='true')
        if len(prob_dict)!=0:
            digraph1.edge(node.id, node.children[0].id, label = f"{prob_dict[node.id][0]:.2f}",fontsize="150pt",penwidth='3')
            digraph1.edge(node.id, node.children[1].id, label = f"{prob_dict[node.id][1]:.2f}",fontsize="150pt",penwidth='3')
        else:
            digraph1.edge(node.id, node.children[0].id, label = f"",penwidth='3')
            digraph1.edge(node.id, node.children[1].id, label = f"",penwidth='3')

        forward_build(node.children[0])
        forward_build(node.children[1])

        return 

    digraph1 = graphviz.Graph(comment="Tree")
    if len(prob_dict)!=0:
        digraph1.node('data', shape='box', image=f"figures/nodes/data.png",label='',imagescale='true', width = '10px', height='10px')
    digraph1.node(model.tree.root.id, shape='box', image=f"figures/nodes/{model.tree.root.id}.png",label='',imagescale='true', width = '10px', height='10px')
    forward_build(node=model.tree.root)

    print(digraph1.source)
    digraph1.render(filename='graph_auto.dot')
    os.system('dot -Tpng -Gsize=15,15\! graph_auto.dot -o figures/output.png')

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Distilling a Neural Network Into a Soft Decision Tree')
    args.add_argument('-c', '--config', default='config_SDT.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, '',run_id='')


    main(config)

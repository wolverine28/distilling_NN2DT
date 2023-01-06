import graphviz
from tree import softTree
from anytree import AnyNode, PreOrderIter, RenderTree
import pickle
import os


if __name__ == '__main__':
    model = softTree(depth = 6, feature_size = 784, n_classes = 10, batch_size = 0)


    with open('prob_dict.pickle', 'rb') as handle:
        prob_dict = pickle.load(handle)



    def forward_build(node):
        if node.is_leaf:    return 
        
        digraph1.node(node.children[0].id   , shape = 'box'
                                            , image = f"figures/{node.children[0].id}.png"
                                            , label = ''
                                            , width = '10px', height='10px',imagescale='true')
        digraph1.node(node.children[1].id   , shape = 'box'
                                            , image = f"figures/{node.children[1].id}.png"
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
        digraph1.node('data', shape='box', image=f"figures/data.png",label='',imagescale='true', width = '10px', height='10px')
    digraph1.node(model.tree.root.id, shape='box', image=f"figures/{model.tree.root.id}.png",label='',imagescale='true', width = '10px', height='10px')
    forward_build(node=model.tree.root)

    print(digraph1.source)
    digraph1.render(filename='graph_auto.dot')
    os.system('dot -Tpng -Gsize=15,15\! graph_auto.dot -o output.png')
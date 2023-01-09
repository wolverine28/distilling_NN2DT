# Distilling a Neural Network into a soft decision tree

Pytorch implementation of ["Distilling a Neural Network into a soft decision tree"](https://arxiv.org/abs/1711.09784) by Nicholas Frosst and Geoffrey Hinton.


# Run the Code

First, run
```
python train_convnet.py
```
to train a LeNet-5 as a Teacher model

Then, run
```
python train_SDT.py
```
to train a soft decision tree

To visualize the trained results run
```
python visualize.py
python create_graph.py
```
The resulting graph will be saved on `figures/output.png`


![output.png]("figures\output.png")
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    Class of Multi-Layer Perceptron which doesn't grow (the number of hidden nodes in each layer is same)
    Inialization Parameters:
     - h_nodes: number of hidden nodes for each layer
     - num_layers: number of internal layers in the MLP
     - in_dim: input dimension of data
     - out_dim: dimension of the resulting output
     - nonlin_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ..)
    """
    def __init__(self, h_nodes=64, num_layers=2, in_dim = 1, out_dim = 1, nonlin_layer = nn.ReLU()):
        super(MLPBlock, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlin_layer = nonlin_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_layers):
            self.layers.append(nn.Linear(h_nodes,h_nodes))
        self.layers.append(nn.Linear(h_nodes, out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)
            # nn.init.zeros_(self.layers[i].bias)

    def forward(self, x):
        # pass input through first layer
        x = self.nonlin_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers):
            x = self.nonlin_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[self.num_layers+1](x)

        return out
    
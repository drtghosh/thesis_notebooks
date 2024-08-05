import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Class of Multi-Layer Perceptron which doesn't grow (the number of hidden nodes in each layer is same)
    Initialization Parameters:
     - h_nodes: number of hidden nodes for each layer
     - num_layers: number of internal layers in the MLP
     - in_dim: input dimension of data
     - out_dim: dimension of the resulting output
     - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, in_dim=1, out_dim=1, nonlinear_layer=nn.ReLU()):
        super(MLPBlock, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinear_layer = nonlinear_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_layers):
            self.layers.append(nn.Linear(h_nodes, h_nodes))
        self.layers.append(nn.Linear(h_nodes, out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, x):
        # pass input through first layer
        x = self.nonlin_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers):
            x = self.nonlin_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[self.num_layers+1](x)

        return out


class MLPGrow(nn.Module):
    """
    Class of Multi-Layer Perceptron which grows by a multiplier
    (the number of hidden nodes in each layer grows exponentially)
    Initialization Parameters:
     - h_nodes: number of hidden nodes for first internal layer
     - num_layers: number of internal layers in the MLP
     - multiplier: the factor by which the number of hidden nodes grows
     - in_dim: input dimension of data
     - out_dim: dimension of the resulting output
     - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, multiplier=2, in_dim=1, out_dim=1, nonlinear_layer=nn.ReLU()):
        super(MLPGrow, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.multiplier = multiplier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinear_layer = nonlinear_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        self.current_h_nodes = h_nodes
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.current_h_nodes, self.current_h_nodes*multiplier))
            self.current_h_nodes *= multiplier
        self.layers.append(nn.Linear(self.current_h_nodes, out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, x):
        # pass input through first layer
        x = self.nonlin_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers):
            x = self.nonlin_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[self.num_layers+1](x)

        return out


class CovNet(nn.Module):
    """
    Network to compute the covariance between two points
    Initialization Parameters:
     - h_nodes: number of hidden nodes for each layer
     - num_layers: number of internal layers in the MLP
     - in_dim: input dimension of data
     - out_dim: dimension of the resulting output
     - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, in_dim=1, out_dim=1, nonlinear_layer=nn.ReLU()):
        super(CovNet, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinear_layer = nonlinear_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_layers):
            self.layers.append(nn.Linear(h_nodes, h_nodes))
        self.layers.append(nn.Linear(h_nodes, out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)

    def individual_pass(self, x):
        # pass input through first layer
        x = self.nonlinear_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers):
            x = self.nonlinear_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        x = self.layers[self.num_layers + 1](x)
        # enforce positivity on the output
        out = nn.Softplus(x)

        return out

    def forward(self, x):
        # compute covariance for original input pair
        cov1 = self.individual_pass(x)
        # collect the dimension of individual input (of pair)
        individual_dim = self.in_dim//2
        # swap the inputs and concatenate the tensors to compute the covariance
        swapped_pair = torch.cat((x[:, :individual_dim], x[:, individual_dim:2*individual_dim]), dim=1)
        cov2 = self.individual_pass(swapped_pair)

        # take an average to ensure symmetry
        return cov1 + cov2


class ResBlock(nn.Module):
    """
        Residual block for the deep residual network
        Initialization Parameters:
         - h_nodes: number of hidden nodes for each layer
         - block_length: number of layers after which residual connection is added
         - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
        """
    def __init__(self, h_nodes=64, block_length=2, nonlinear_layer=torch.nn.ReLU()):
        super(ResBlock, self).__init__()
        self.h_nodes = h_nodes
        self.block_length = block_length
        self.nonlinear_layer = nonlinear_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        for i in range(block_length):
            self.layers.append(nn.Linear(h_nodes, h_nodes))

        # initialize the weights
        for i in range(block_length):
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, x):
        # store input
        x_in = x
        # pass through the layers
        for i in range(self.block_length):
            x = self.nonlinear_layer(self.layers[i](x))
        # add residual
        out = x_in + x
        return out


class ResDeepBlock(nn.Module):
    """
        Deep network with residual connections
        Initialization Parameters:
         - h_nodes: number of hidden nodes for each layer
         - num_blocks: number of residual blocks in the network
         - block_length: length (number of layers) of each residual connection
         - in_dim: input dimension of data
         - out_dim: dimension of the resulting output
         - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
        """
    def __init__(self, h_nodes=64, num_blocks=2, block_length=2, in_dim=1, out_dim=1, nonlinear_layer=nn.ReLU()):
        super(ResDeepBlock, self).__init__()
        self.h_nodes = h_nodes
        self.num_blocks = num_blocks
        self.block_length = block_length
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinear_layer = nonlinear_layer

        # creating the layers with internal residual blocks
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_blocks):
            self.layers.append(ResBlock(h_nodes=h_nodes, block_length=block_length, nonlinear_layer=nonlinear_layer))
        self.layers.append(nn.Linear(h_nodes, out_dim))

        # initialize the weights of non-residual layers
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[len(self.layers)-1].weight)

    def forward(self, x):
        # pass input through first layer
        x = self.nonlin_layer(self.layers[0](x))
        # pass through internal residual blocks
        for i in range(1, len(self.layers)-1):
            x = self.layers[i](x)
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[len(self.layers)-1](x)

        return out

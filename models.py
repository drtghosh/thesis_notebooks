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
     - nonlin_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, in_dim=1, out_dim=1, nonlin_layer=nn.ReLU()):
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
            self.layers.append(nn.Linear(h_nodes, h_nodes))
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
     - nonlin_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, multiplier=2, in_dim=1, out_dim=1, nonlin_layer=nn.ReLU()):
        super(MLPGrow, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.multiplier = multiplier
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlin_layer = nonlin_layer

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

class covNet(nn.Module):
    """
    Network to compute the covariance between two points
    Initialization Parameters:
     - h_nodes: number of hidden nodes for each layer
     - num_layers: number of internal layers in the MLP
     - in_dim: input dimension of data
     - out_dim: dimension of the resulting output
     - nonlin_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, h_nodes=64, num_layers=2, in_dim=1, out_dim=1, nonlin_layer=nn.ReLU()):
        super(covNet, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.m = nonlin_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_layers):
            self.layers.append(nn.Linear(h_nodes, h_nodes))
        self.layers.append(nn.Linear(h_nodes, out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)
            # nn.init.zeros_(self.layers[i].bias)

    def individual_pass(self, x):
        # pass input through first layer
        x = self.nonlin_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers):
            x = self.nonlin_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        x = self.layers[self.num_layers + 1](x)
        # enforce positivity on the output
        out = nn.Softplus(x)

        return out

    def forward(self, x):
        out1 = self.individual_pass(x)
        # Swap the first dim dimensions with the second dim dimensions
        dim = x.shape[1]//2
        first_dim_columns = x[:, :dim]
        second_dim_columns = x[:, dim:]
        # Swap the columns and concatenate the tensors
        swapped_input = torch.cat((second_dim_columns, first_dim_columns), dim=1)
        out2 = self.individual_pass(swapped_input)

        # Swap the columns back
        return out1 + out2


class deep_ritz_block(nn.Module):
    def __init__(self, k=64, device=None, m=torch.nn.ReLU()):
        super(deep_ritz_block, self).__init__()
        self.k = k
        self.device = device
        self.m = m
        self.first_layer = nn.Linear(k, k).to(device)
        nn.init.xavier_uniform_(self.first_layer.weight)
        self.second_layer = nn.Linear(k, k).to(device)
        nn.init.xavier_uniform_(self.second_layer.weight)

    def forward(self, inputs):
        # out = inputs.to(self.device)
        out = self.first_layer(inputs)
        out = self.m(out)
        out = self.second_layer(out)
        out = self.m(out)
        out = out + inputs
        return out


class deep_ritz_network(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, k=64, device=None, m=torch.nn.ReLU(), num_blocks=2):
        super(deep_ritz_network, self).__init__()
        self.k = k
        self.device = device
        self.m = m
        self.num_blocks = num_blocks
        self.input_dim = input_dim
        self.first_layer = nn.Linear(input_dim, k).to(device)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(deep_ritz_block(k=k, device=device, m=m))
        self.last_layer = nn.Linear(k, output_dim).to(device)
        nn.init.xavier_uniform_(self.first_layer.weight)
        nn.init.xavier_uniform_(self.last_layer.weight)

    def forward(self, inputs):
        out = self.first_layer(inputs)
        # out = self.m(out)
        for i in range(self.num_blocks):
            out = self.blocks[i](out)
        out = self.last_layer(out)
        return out

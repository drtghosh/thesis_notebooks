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


class TNet(nn.Module):
     """
        For learning a Transformation matrix with a specified dimension
     """
    def __init__(self, dim, num_points):
        super(TNet, self).__init__()

        # dimensions for transformation matrix
        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim ** 2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]

        # forward through conv1d (shared MLP) layers
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over number of input points
        x = self.max_pool(x).view(bs, -1)

        # pass through fully connected (linear) layers
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x


class PointNetEncoder(nn.Module):
    def __init__(self, input_points, input_dim, dim_global_feature):
        ''' Initializers:
                input_points - number of points in input point cloud
                input_dim - dimension of input point space
                dim_global_feature - dimension of Global Features for the main
                                   Max Pooling layer
            '''
        super(PointNetEncoder, self).__init__()

        self.input_points = input_points
        self.input_dim = input_dim
        self.dim_global_feature = dim_global_feature

        # TNets
        self.tnet_input = TNet(dim=input_dim, num_points=input_points)
        self.tnet_inter = TNet(dim=64, num_points=input_points)

        # conv1d (shared MLP) before feature transform
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # conv1d (shared MLP) after feature transform
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.dim_global_feature, kernel_size=1)

        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.dim_global_feature)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=input_points, return_indices=True)

    def forward(self, x):
        # get batch size
        bs = x.shape[0]

        # pass through first Tnet to get transformation matrix
        T_input = self.tnet_input(x)

        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), T_input).transpose(2, 1)

        # pass through first 1d convnets
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        # compute feature transform
        T_feature = self.tnet_inter(x)

        # perform second transformation across each (64 dim) feature in the batch
        x = torch.bmm(x.transpose(2, 1), T_feature).transpose(2, 1)

        # pass through second set of 1d convnets
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        return global_features, critical_indexes, T_feature

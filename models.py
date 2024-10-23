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
        x = self.nonlinear_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers+1):
            x = self.nonlinear_layer(self.layers[i](x))
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
        x = self.nonlinear_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers+1):
            x = self.nonlinear_layer(self.layers[i](x))
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
        positive_nonlinear = nn.Softplus()
        out = positive_nonlinear(x)

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
        x = self.nonlinear_layer(self.layers[0](x))
        # pass through internal residual blocks
        for i in range(1, len(self.layers)-1):
            x = self.layers[i](x)
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[len(self.layers)-1](x)

        return out


class TNet(nn.Module):
    """
        For learning a Transformation matrix with a specified dimension
        to be used in PointNet architecture
        Initialization Parameters:
         - in_dim: dimension of the input point space
         - h_nodes: number of hidden nodes for first shared layer
         - multiplier: the factor by which the number of hidden nodes grows
         - shared_layers: number of shared MLP layers (easier to use 1D convolution for sharing)
         - linear_layers: number of fully connected layers at the end
         - num_points: number of points in the input point cloud
         - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, in_dim=3, h_nodes=64, multiplier=2, shared_layers=3, linear_layers=3, num_points=1024, nonlinear_layer=nn.ReLU()):
        super(TNet, self).__init__()
        self.in_dim = in_dim
        self.h_nodes = h_nodes
        self.multiplier = multiplier
        self.shared_layers = shared_layers
        self.linear_layers = linear_layers
        self.num_points = num_points
        self.nonlinear_layer = nonlinear_layer

        # creating the network layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        # shared layers (1D convolutions)
        self.layers.append(nn.Conv1d(in_dim, h_nodes, 1))
        for i in range(1, shared_layers):
            self.layers.append(nn.Conv1d(h_nodes*(multiplier**(i-1)), h_nodes*(multiplier**i), 1))
            self.bn_layers.append(nn.BatchNorm1d(h_nodes*(multiplier**(i-1))))
        # get the last layer hidden node count
        self.final_width = h_nodes*(multiplier**(shared_layers-1))
        self.bn_layers.append(nn.BatchNorm1d(self.final_width))
        for j in range(linear_layers-1):
            self.layers.append(nn.Linear(self.final_width//multiplier**j, self.final_width//multiplier**(j+1)))
            self.bn_layers.append(nn.BatchNorm1d(self.final_width//multiplier**(j+1)))
        self.final_width = self.final_width//multiplier**(linear_layers-1)
        self.layers.append(nn.Linear(self.final_width, in_dim**2))
        # max pooling of each point
        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        # get batch size
        bs = x.shape[0]
        # forward through shared layers
        for i in range(self.shared_layers):
            x = self.bn_layers[i](self.nonlinear_layer(self.layers[i](x)))
        # max pool over number of input points
        x = self.max_pool(x).view(bs, -1)
        # pass through fully connected (linear) layers
        for j in range(self.linear_layers):
            if j < self.linear_layers-1:
                x = self.bn_layers[self.shared_layers+j](self.nonlinear_layer(self.layers[self.shared_layers+j](x)))
            else:
                x = self.layers[self.shared_layers+j](x)
        # initialize identity matrix
        iden = torch.eye(self.in_dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        out = x.view(-1, self.in_dim, self.in_dim) + iden

        return out


class PointNetEncoder(nn.Module):
    """
        PointNet encoder for point cloud data
        Initialization Parameters:
         - in_points: number of input points in the point cloud
         - in_dim: dimension of the input point space
         - shared_before: number of shared layers before feature transform
         - feature_dim: dimension of the feature space
         - shared_after: number of shared layers after feature transform
         - multiplier: the factor by which the number of hidden nodes grows
         - global_feature_dim: dimension of the encoding
         - nonlinear_layer: non-linearity added to the network layers (e.g. ReLU, softmax, ...)
    """
    def __init__(self, in_points, in_dim, shared_before, feature_dim, shared_after, multiplier, global_feature_dim, nonlinear_layer=nn.ReLU()):
        super(PointNetEncoder, self).__init__()
        self.in_points = in_points
        self.in_dim = in_dim
        self.shared_before = shared_before
        self.feature_dim = feature_dim
        self.shared_after = shared_after
        self.multiplier = multiplier
        self.global_feature_dim = global_feature_dim
        self.nonlinear_layer = nonlinear_layer

        # initialize TNets
        self.tnet_input = TNet(in_dim=in_dim, num_points=in_points)
        self.tnet_inter = TNet(in_dim=feature_dim, num_points=in_points)
        # creating the network layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        # shared layers (1D convolution) before feature transform
        self.layers.append(nn.Conv1d(in_dim, feature_dim, 1))
        self.bn_layers.append(nn.BatchNorm1d(feature_dim))
        for i in range(1, shared_before):
            self.layers.append(nn.Conv1d(feature_dim, feature_dim, 1))
            self.bn_layers.append(nn.BatchNorm1d(feature_dim))
        # shared layers (1D convolution) after feature transform
        for j in range(shared_after-1):
            self.layers.append(nn.Conv1d(feature_dim*(multiplier**j), feature_dim*(multiplier**(j+1)), 1))
            self.bn_layers.append(nn.BatchNorm1d(feature_dim*(multiplier**(j+1))))
        self.layers.append(nn.Conv1d(feature_dim*(multiplier**(self.shared_after-1)), global_feature_dim, 1))
        self.bn_layers.append(nn.BatchNorm1d(global_feature_dim))
        # max pooling of each point for the global features
        self.max_pool = nn.MaxPool1d(kernel_size=in_points, return_indices=True)

    def forward(self, x):
        # get batch size
        bs = x.shape[0]
        # pass through first Tnet to get transformation matrix
        t_input = self.tnet_input(x)
        # perform first transformation across each point in the batch
        x = torch.bmm(x.transpose(2, 1), t_input).transpose(2, 1)
        # pass through first 1d convnets
        for i in range(self.shared_before):
            x = self.bn_layers[i](self.nonlinear_layer(self.layers[i](x)))
        # compute feature transform
        t_feature = self.tnet_inter(x)
        # perform second transformation across each feature in the batch
        x = torch.bmm(x.transpose(2, 1), t_feature).transpose(2, 1)
        # pass through second set of 1d convnets
        for j in range(self.shared_after):
            x = self.bn_layers[self.shared_before+j](self.nonlinear_layer(self.layers[self.shared_before+j](x)))
        # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        # critical_indexes = critical_indexes.view(bs, -1)

        return global_features  # critical_indexes, t_feature


class DeepGCN(nn.Module):
    def __init__(self, h_nodes=64, num_layers=1, data_dim=1, in_dim=1, num_kernels=1, nonlinear_layer=nn.ReLU()):
        super(DeepGCN, self).__init__()
        self.h_nodes = h_nodes
        self.num_layers = num_layers
        self.data_dim = data_dim
        self.in_dim = in_dim
        self.num_kernels = num_kernels
        self.out_dim = data_dim * num_kernels
        self.nonlinear_layer = nonlinear_layer

        # creating the linear (fully connected) layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, h_nodes))
        for i in range(num_layers):
            self.layers.append(nn.Linear(h_nodes, h_nodes))
        self.layers.append(nn.Linear(h_nodes, self.out_dim))

        # initialize the weights
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, x):
        # pass input through first layer
        x = self.nonlinear_layer(self.layers[0](x))
        # pass through internal layers
        for i in range(1, self.num_layers+1):
            x = self.nonlinear_layer(self.layers[i](x))
        # pass through last layer to get output (no activation on the last layer)
        out = self.layers[self.num_layers+1](x)

        return out.reshape(-1, self.data_dim, self.num_kernels)

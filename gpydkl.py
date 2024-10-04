# import numpy as np
import torch
import gpytorch
import numpy as np
from create_data import DumbCirc as dC
from scipy.stats import norm
# from torch.distributions import MultivariateNormal as multiNorm
import matplotlib.pyplot as plt
# import torch.nn as nn
# import time
# import sys
# import os
# import gpytoolbox
from models import MLPGrow, PointNetEncoder


# from itertools import cycle
# import matplotlib.pyplot as plt
# import pickle
# import scipy
# import multiprocessing


class DKLGPy:
    """
    Class of Neural Shape Uncertainty Quantification conditioned on a given incomplete point cloud
    by minimizing the -ve log likelihood of the complete point cloud data estimate from the posterior
    of Gaussian Process!
    Initialization Parameters:
     - space_dim: dimension of the input point space
     - point_cloud: complete point cloud dataset
     - partial_cloud: incomplete point cloud dataset
     - partial_value: distance value of the observed partial points
     - noise_present: boolean for whether noise is present in the partially observed data
     - noise_var: amount of noise to be considered in case noise_present is True
     - test_partial: test data of partial point cloud to predict on
     - latent_dim: dimension of the generated latent code for partial point cloud
     - cov_layers: number of internal layers for the covariance network
     - hidden_nodes: number of hidden nodes in the covariance network inner layers
     - device: use 'cuda' if available
    """

    def __init__(self, space_dim=2, point_cloud=None, partial_cloud=None, partial_value_train=None, train_labels=None,
                 noise_present=True, noise_var=0.01, test_partial=None, partial_value_test=None, test_labels=None,
                 latent_dim=256, cov_layers=3, hidden_nodes=64, mapping_dim=256 + 256, grid_size=100, count_labels=0,
                 negative_cloud=None, device=None):
        self.space_dim = space_dim
        self.latent_dim = latent_dim
        self.mapping_dim = mapping_dim
        self.cov_layers = cov_layers
        self.grid_sizes = np.ones(self.space_dim, dtype=np.int32) * grid_size
        self.count_labels = count_labels
        self.negative_cloud = negative_cloud
        self.device = device
        self.point_cloud = point_cloud
        self.partial_cloud = partial_cloud
        self.partial_value_train = partial_value_train
        self.train_labels = train_labels
        self.partial_value_test = partial_value_test
        self.test_labels = test_labels
        self.noise_present = noise_present
        if noise_present:
            self.noise_var = noise_var
        self.test_partial = test_partial

        # set point cloud data or assert that one of point cloud data and space dim is specified
        if point_cloud is not None:
            self.set_training_data(point_cloud, partial_cloud, negative_cloud, train_labels)
        else:
            assert space_dim is not None
            self.space_dim = space_dim

        # initialize the encoder network
        self.encoder = PointNetEncoder(self.partial_cloud.shape[1], self.space_dim, 2, 64, 3, 2, self.latent_dim)

        # initialize the mapping network for covariance
        if count_labels:
            new_in_dim = self.space_dim + self.count_labels
        else:
            new_in_dim = self.space_dim + self.latent_dim
        self.map_network = MLPGrow(h_nodes=hidden_nodes, num_layers=cov_layers, in_dim=new_in_dim, out_dim=mapping_dim)

        # conditioned point cloud size
        conditional_size = self.space_dim + self.latent_dim

        # covariance kernel
        self.covar_module_data = gpytorch.kernels.RBFKernel(ard_num_dims=self.space_dim).to(self.device)
        self.covar_module_conditioned = gpytorch.kernels.RBFKernel(ard_num_dims=conditional_size).to(self.device)
        self.covar_after_mapping = gpytorch.kernels.RBFKernel(ard_num_dims=mapping_dim).to(self.device)

        # normalization layer
        self.data_norm = torch.nn.BatchNorm1d(num_features=self.space_dim)
        self.enc_norm = torch.nn.BatchNorm1d(num_features=conditional_size, affine=False)
        self.map_norm = torch.nn.BatchNorm1d(num_features=mapping_dim)

        # for combined kernel
        self.alpha = torch.tensor(0.0)

    def set_device(self, device):
        self.device = device
        self.point_cloud.to(self.device)
        self.partial_cloud.to(self.device)
        self.partial_value_train.to(self.device)
        self.encoder.to(self.device)
        self.map_network.to(self.device)

    def set_training_data(self, point_cloud, partial_cloud, negative_cloud, train_labels, partial_value_train=None):
        assert point_cloud.shape[0] == partial_cloud.shape[0]
        assert point_cloud.shape[2] == partial_cloud.shape[2]
        assert point_cloud.shape[1] >= partial_cloud.shape[1]
        assert point_cloud.shape[0] == negative_cloud.shape[0]
        assert point_cloud.shape[2] == negative_cloud.shape[2]
        assert point_cloud.shape[1] == negative_cloud.shape[1]
        self.point_cloud = point_cloud
        self.space_dim = point_cloud.size(-1)
        self.partial_cloud = partial_cloud
        self.negative_cloud = negative_cloud
        if partial_value_train is not None:
            self.partial_value_train = partial_value_train
        else:
            if self.noise_present:
                self.partial_value_train = self.noise_var * torch.randn(self.partial_cloud.size()[:-1])
            else:
                self.partial_value_train = torch.zeros(self.partial_cloud.size()[:-1])
        self.train_labels = train_labels
        # self.fpc_copy = torch.tensor(point_cloud, dtype=torch.float32, device=self.device)
        # self.ppc_copy = torch.tensor(partial_cloud, dtype=torch.float32, device=self.device)
        # self.fpc_repeated = torch.cat((self.fpc_copy, self.fpc_copy), dim=1).to(self.device)

    def set_test_data(self, test_partial, test_labels, partial_value_test=None):
        self.test_partial = test_partial
        self.test_labels = test_labels
        if partial_value_test is not None:
            self.partial_value_test = partial_value_test
        else:
            if self.noise_present:
                self.partial_value_test = self.noise_var * torch.randn(self.test_partial.size()[:-1])
            else:
                self.partial_value_test = torch.zeros(self.test_partial.size()[:-1])
        self.test_partial.to(self.device)
        self.partial_value_test.to(self.device)

    def get_gp_data(self):
        # train_x = torch.cat((self.partial_cloud, self.point_cloud), 1).to(self.device)
        train_x = self.partial_cloud.to(self.device)
        if self.count_labels:
            x = torch.cat((train_x, self.train_labels.unsqueeze(1).repeat(1, train_x.size(1), 1)), 2)
        else:
            # compute encoding in a batch
            encoding = self.encoder(self.partial_cloud.transpose(1, 2))
            # repeat encoding for each point in full point cloud
            x = torch.cat((train_x, encoding.unsqueeze(1).repeat(1, train_x.size(1), 1)), 2)
        x = x.reshape(-1, x.size(-1))
        y = self.partial_value_train.reshape(-1)
        return x, y

    def create_grid(self, box_min=None, box_max=None, eps=0.1):
        # find the bounding box for all dataset
        if box_min is None:
            box_min = torch.amin(self.test_partial, (1, 0)) - eps
        if box_max is None:
            box_max = torch.amax(self.test_partial, (1, 0)) + eps

        # Build a grid (dimension-agnostic)
        grid_vertices = np.meshgrid(
            *[np.linspace(box_min[d], box_max[d], self.grid_sizes[d]) for d in range(self.space_dim)])
        grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, self.space_dim)
        grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
        return grid_vertices

    def get_test_data(self):
        points_to_predict = self.create_grid()
        # combine all data
        test_x = points_to_predict.repeat(self.test_partial.size(0), 1, 1).to(self.device)
        # add label information if required
        if self.count_labels:
            test_x = torch.cat((test_x, self.test_labels.unsqueeze(1).repeat(1, test_x.size(1), 1)), 2)
        else:
            # collect partial data
            partial = self.test_partial.to(self.device)
            # set the encoder to evaluation mode
            self.encoder.eval()
            # compute encoding in a batch
            encoding = self.encoder(partial.transpose(1, 2))
            # repeat encoding for each point in full point cloud
            test_x = torch.cat((test_x, encoding.unsqueeze(1).repeat(1, test_x.size(1), 1)), 2)

        return test_x

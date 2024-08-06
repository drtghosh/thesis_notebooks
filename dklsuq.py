# import numpy as np
import torch
# import torch.nn as nn
# import time
# import sys
# import os
# import gpytoolbox
from .models import MLPGrow
# from itertools import cycle
# import matplotlib.pyplot as plt
# import pickle
# import scipy
# import multiprocessing


class DeepKernelSUQ:
    """
    Class of Neural Shape Uncertainty Quantification conditioned on a given incomplete point cloud
    by minimizing the -ve log likelihood of the complete point cloud data estimate from the posterior
    of Gaussian Process!
    Initialization Parameters:
     - space_dim: dimension of the input point space
     - point_cloud: complete point cloud dataset
     - partial_cloud: incomplete point cloud dataset
     - latent_dim: dimension of the generated latent code for partial point cloud
     - cov_layers: number of internal layers for the covariance network
     - hidden_nodes: number of hidden nodes in the covariance network inner layers
     - device: use 'cuda' if available
    """
    def __init__(self, space_dim=2, point_cloud=None, partial_cloud=None, latent_dim=1024, cov_layers=5, hidden_nodes=64, device=None):
        self.space_dim = space_dim
        self.latent_dim = latent_dim
        self.cov_layers = cov_layers
        self.device = device

        # set point cloud data or assert that one of point cloud data and space dim is specified
        if point_cloud is not None:
            self.set_data(point_cloud, partial_cloud)
        else:
            assert space_dim is not None
            self.space_dim = space_dim

        # initialize the covariance network
        self.cov_network = MLPGrow(h_nodes=hidden_nodes, num_layers=cov_layers, in_dim=(self.space_dim + self.latent_dim), out_dim=1, nonlinear_layer=torch.sin)
        self.cov_network.to(device)

    def set_data(self, point_cloud, partial_cloud):
        assert point_cloud.shape[0] == partial_cloud.shape[0]
        assert point_cloud.shape[2] == partial_cloud.shape[2]
        assert point_cloud.shape[1] >= partial_cloud.shape[1]
        self.point_cloud = point_cloud
        self.space_dim = point_cloud.shape[1]
        self.partial_cloud = partial_cloud
        self.fpc_copy = torch.tensor(point_cloud, dtype=torch.float32, device=self.device)
        self.ppc_copy = torch.tensor(partial_cloud, dtype=torch.float32, device=self.device)
        self.fpc_repeated = torch.cat((self.fpc_copy, self.fpc_copy), dim=1).to(self.device)

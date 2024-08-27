# import numpy as np
import torch
import torch.nn as nn
# import time
# import sys
# import os
# import gpytoolbox
from models import CovNet, ResDeepBlock, PointNetEncoder


# from itertools import cycle
# import matplotlib.pyplot as plt
# import pickle
# import scipy
# import multiprocessing


class NeuralSUQ:
    """
    Class of Neural Shape Uncertainty Quantification conditioned on a given incomplete point cloud
    by minimizing the -ve log likelihood of the complete point cloud data estimate from the posterior
    of Gaussian Process!
    Initialization Parameters:
     - space_dim: dimension of the input point space
     - point_cloud: complete point cloud dataset
     - partial_cloud: incomplete point cloud dataset
     - partial_value: distance value of the observed partial points
     - noise present: boolean for whether noise is present in the partially observed data
     - test_partial: test data of partial point cloud to predict on
     - latent_dim: dimension of the generated latent code for partial point cloud
     - cov_layers: number of internal layers for the covariance network
     - hidden_nodes: number of hidden nodes in the covariance network inner layers
     - add_residual: boolean to decide whether to use residual blocks or not
     - device: use 'cuda' if available
    """

    def __init__(self, space_dim=2, point_cloud=None, partial_cloud=None, partial_value=None, noise_present=True,
                 test_partial=None, latent_dim=1024, cov_layers=5, hidden_nodes=256, add_residual=False, device=None):
        self.space_dim = space_dim
        self.latent_dim = latent_dim
        self.cov_layers = cov_layers
        self.device = device
        self.point_cloud = point_cloud
        self.partial_cloud = partial_cloud
        self.partial_value = partial_value
        self.noise_present = noise_present
        self.test_partial = test_partial

        # set point cloud data or assert that one of point cloud data and space dim is specified
        if point_cloud is not None:
            self.set_training_data(point_cloud, partial_cloud)
        else:
            assert space_dim is not None
            self.space_dim = space_dim

        # initialize the encoder network
        self.encoder = PointNetEncoder(self.partial_cloud.shape[1], self.space_dim, 2, 64, 3, 2, 1024)

        # initialize the covariance network
        if add_residual:
            self.cov_network = nn.Sequential(
                ResDeepBlock(h_nodes=hidden_nodes, in_dim=(2 * self.space_dim + self.latent_dim), out_dim=1,
                             nonlinear_layer=torch.sin), nn.Softplus())
        else:
            self.cov_network = CovNet(h_nodes=hidden_nodes, num_layers=cov_layers,
                                      in_dim=(2 * self.space_dim + self.latent_dim), out_dim=1,
                                      nonlinear_layer=torch.sin)

    def set_device(self, device):
        self.device = device
        self.point_cloud.to(self.device)
        self.partial_cloud.to(self.device)
        self.partial_value.to(self.device)
        self.encoder.to(self.device)
        self.cov_network.to(self.device)

    def set_training_data(self, point_cloud, partial_cloud, partial_value=None):
        assert point_cloud.shape[0] == partial_cloud.shape[0]
        assert point_cloud.shape[2] == partial_cloud.shape[2]
        assert point_cloud.shape[1] >= partial_cloud.shape[1]
        self.point_cloud = point_cloud
        self.space_dim = point_cloud.size(-1)
        self.partial_cloud = partial_cloud
        if partial_value is not None:
            self.partial_value = partial_value
        else:
            if self.noise_present:
                self.partial_value = 0.01 * torch.randn(self.partial_cloud.size()[:-1])
            else:
                self.partial_value = torch.zeros(self.partial_cloud.size()[:-1])
        # self.fpc_copy = torch.tensor(point_cloud, dtype=torch.float32, device=self.device)
        # self.ppc_copy = torch.tensor(partial_cloud, dtype=torch.float32, device=self.device)
        # self.fpc_repeated = torch.cat((self.fpc_copy, self.fpc_copy), dim=1).to(self.device)

    def set_test_data(self, test_partial):
        self.test_partial = test_partial
        self.test_partial.to(self.device)

    def get_posterior(self, x):
        partial = x[:, self.point_cloud.shape[1]:, :]
        partial = partial.to(self.device)
        # compute encoding in a batch
        encoding = self.encoder(partial.transpose(1, 2))
        # collect batch size
        bs = x.size(0)
        # create empty list to store multivariate normals
        posterior_nlls = torch.empty(bs).to(self.device)
        # compute point cloud possible combinations
        c = torch.combinations(torch.arange(x.size(1)), r=2, with_replacement=True)
        # repeat data for filtering
        x_rep = x[:, None, :].expand(-1, len(c), -1, -1)
        # create data with encoding
        for i in range(bs):
            # print(f'cloud {i}')
            cov_x = torch.empty((len(c), 2 * self.space_dim + self.latent_dim)).to(self.device)
            cov_matrix = torch.empty((x.size(1), x.size(1))).to(self.device)
            m, n = torch.triu_indices(x.size(1), x.size(1))
            for j in range(len(c)):
                cov_x[j] = torch.concat((x_rep[i][j][c[j]].flatten(), encoding[i]))
            cov_entries = self.cov_network(cov_x).flatten()
            cov_matrix[m, n] = cov_entries
            cov_matrix.T[m, n] = cov_entries
            kernel_ff = cov_matrix[:self.point_cloud.shape[1], :self.point_cloud.shape[1]]
            kernel_pf = cov_matrix[self.point_cloud.shape[1]:, :self.point_cloud.shape[1]]
            kernel_pp = cov_matrix[self.point_cloud.shape[1]:, self.point_cloud.shape[1]:]
            posterior_mean = kernel_pf.T @ torch.linalg.inv(kernel_pp) @ self.partial_value[i].to(self.device)
            posterior_var = kernel_ff - kernel_pf.T @ torch.linalg.inv(kernel_pp) @ kernel_pf
            # posterior_nlls[i] = -torch.distributions.MultivariateNormal(posterior_mean, posterior_var).log_prob(full[i])
            posterior_nlls[i] = 0.5 * (torch.log(torch.linalg.det(posterior_var) + 1e-6) - torch.log(torch.tensor(1e-6))
                                       + posterior_mean.T @ torch.linalg.inv(posterior_var) @ posterior_mean)

        return posterior_nlls.to(self.device)

    def train(self, num_epochs=200, print_every=1, learning_rate=0.001, weight_decay=1e-5):
        train_x = torch.cat((self.point_cloud, self.partial_cloud), 1).to(self.device)
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.cov_network.parameters()}
        ], learning_rate, weight_decay=weight_decay)

        for i in range(num_epochs):
            # print(f'Epoch {i}:')
            optimizer.zero_grad()
            output = self.get_posterior(train_x)
            # print(output)
            loss = torch.mean(output)
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                print("Epoch: %d, Loss: %f" % (i, loss.item()))

    # def predict(self, partial, points_to_predict):
    def create_grid(self):
        # find the bounding box for all dataset
        test_x = torch.tensor(self.test_partial).to(self.device)

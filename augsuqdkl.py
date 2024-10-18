# import numpy as np
import torch
import gpytorch
import numpy as np
import torch.nn.functional as fn
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


class AugmentedSUQ:
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
        if count_labels:
            conditional_size = self.space_dim + count_labels
        else:
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

    def get_posterior_zero(self, x, y, labels):
        if self.count_labels:
            x = torch.cat((x, labels.unsqueeze(1).repeat(1, x.size(1), 1)), 2)
        else:
            partial = x[:, :self.partial_cloud.shape[1], :]
            partial = partial.to(self.device)
            # compute encoding in a batch
            encoding = self.encoder(partial.transpose(1, 2))
            # repeat encoding for each point in full point cloud
            x = torch.cat((x, encoding.unsqueeze(1).repeat(1, x.size(1), 1)), 2)
        # collect batch size
        bs = x.size(0)
        # create empty list to store negative log-likelihoods of multivariate normals
        posterior_nlls = torch.empty(bs).to(self.device)
        for i in range(bs):
            mapping = self.map_norm(self.map_network(x[i]))
            cov_matrix_conditioned = self.covar_module_conditioned(self.enc_norm(x[i])).evaluate_kernel().to_dense().to(
                self.device)
            cov_matrix_mapping = self.covar_after_mapping(mapping).evaluate_kernel().to_dense().to(self.device)
            cov_matrix = self.alpha * cov_matrix_conditioned + (1-self.alpha) * cov_matrix_mapping
            # cov_matrix = self.covar_after_mapping(mapping).evaluate_kernel().to_dense().to(self.device)
            kernel_pp = cov_matrix[:self.partial_cloud.shape[1], :self.partial_cloud.shape[1]]
            kernel_ff_pos = cov_matrix[
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1],
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1]]
            kernel_ff_neg = cov_matrix[self.partial_cloud.shape[1] + self.point_cloud.shape[1]:,
                            self.partial_cloud.shape[1] + self.point_cloud.shape[1]:]
            kernel_pf_pos = cov_matrix[:self.partial_cloud.shape[1],
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1]]
            kernel_pf_neg = cov_matrix[:self.partial_cloud.shape[1],
                            self.partial_cloud.shape[1] + self.point_cloud.shape[1]:]
            additional_noise = self.noise_var * torch.eye(self.partial_cloud.shape[1]).to(self.device)
            kernel_with_noise = (kernel_pp + additional_noise).to(self.device)
            posterior_mean_pos = kernel_pf_pos.T @ torch.linalg.inv(kernel_with_noise) @ y[i]
            posterior_var_pos = kernel_ff_pos - kernel_pf_pos.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf_pos
            posterior_mean_neg = kernel_pf_neg.T @ torch.linalg.inv(kernel_with_noise) @ y[i]
            posterior_var_neg = kernel_ff_neg - kernel_pf_neg.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf_neg
            posterior_nlls[i] = 0.5 * (
                    torch.log(torch.linalg.det(posterior_var_pos) + 1e-6) - torch.log(torch.tensor(1e-6))
                    - torch.log(torch.linalg.det(posterior_var_neg) + 1e-6) + torch.log(torch.tensor(1e-6))
                    + posterior_mean_pos.T @ torch.linalg.inv(posterior_var_pos) @ posterior_mean_pos
                    - posterior_mean_neg.T @ torch.linalg.inv(posterior_var_neg) @ posterior_mean_neg)

        return posterior_nlls.to(self.device)

    def get_posterior_binary(self, x, y, labels, reg_const=1.0):
        if self.count_labels:
            x = torch.cat((x, labels.unsqueeze(1).repeat(1, x.size(1), 1)), 2)
        else:
            partial = x[:, :self.partial_cloud.shape[1], :]
            partial = partial.to(self.device)
            # compute encoding in a batch
            encoding = self.encoder(partial.transpose(1, 2))
            # repeat encoding for each point in full point cloud
            x = torch.cat((x, encoding.unsqueeze(1).repeat(1, x.size(1), 1)), 2)
        # collect batch size
        bs = x.size(0)
        # create empty list to store negative log-likelihoods of multivariate normals
        posterior_loss = torch.empty(bs).to(self.device)
        for i in range(bs):
            mapping = self.map_norm(self.map_network(x[i]))
            cov_matrix_conditioned = self.covar_module_conditioned(self.enc_norm(x[i])).evaluate_kernel().to_dense().to(
                self.device)
            cov_matrix_mapping = self.covar_after_mapping(mapping).evaluate_kernel().to_dense().to(self.device)
            cov_matrix = self.alpha * cov_matrix_conditioned + (1-self.alpha) * cov_matrix_mapping
            # cov_matrix = self.covar_after_mapping(mapping).evaluate_kernel().to_dense().to(self.device)
            kernel_pp = cov_matrix[:self.partial_cloud.shape[1], :self.partial_cloud.shape[1]]
            kernel_ff_pos = cov_matrix[
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1],
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1]]
            kernel_ff_neg = cov_matrix[self.partial_cloud.shape[1] + self.point_cloud.shape[1]:,
                            self.partial_cloud.shape[1] + self.point_cloud.shape[1]:]
            kernel_pf_pos = cov_matrix[:self.partial_cloud.shape[1],
                            self.partial_cloud.shape[1]:self.partial_cloud.shape[1] + self.point_cloud.shape[1]]
            kernel_pf_neg = cov_matrix[:self.partial_cloud.shape[1],
                            self.partial_cloud.shape[1] + self.point_cloud.shape[1]:]
            additional_noise = self.noise_var * torch.eye(self.partial_cloud.shape[1]).to(self.device)
            kernel_with_noise = (kernel_pp + additional_noise).to(self.device)
            posterior_mean_pos = kernel_pf_pos.T @ torch.linalg.inv(kernel_with_noise) @ y[i]
            posterior_var_pos = kernel_ff_pos - kernel_pf_pos.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf_pos
            posterior_mean_neg = kernel_pf_neg.T @ torch.linalg.inv(kernel_with_noise) @ (y[i] + 1)
            posterior_var_neg = kernel_ff_neg - kernel_pf_neg.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf_neg
            loss_nll = 0.5 * (
                    torch.log(torch.linalg.det(posterior_var_pos) + 1e-6) - torch.log(torch.tensor(1e-6))
                    + torch.log(torch.linalg.det(posterior_var_neg) + 1e-6) - torch.log(torch.tensor(1e-6))
                    + posterior_mean_pos.T @ torch.linalg.inv(posterior_var_pos) @ posterior_mean_pos
                    + (posterior_mean_neg - 1).T @ torch.linalg.inv(posterior_var_neg) @ (posterior_mean_neg - 1))
            loss_entropy = - ((fn.softmax(cov_matrix, dim=1) * fn.log_softmax(cov_matrix, dim=1)).sum(dim=1)).mean()
            # print(loss_entropy)
            posterior_loss[i] = loss_nll + reg_const * loss_entropy

        return posterior_loss.to(self.device)

    def train(self, num_epochs=20, batch_size=16, print_every=1, learning_rate=0.005, weight_decay=1e-5, kind='z'):
        train_x = torch.cat((self.partial_cloud, self.point_cloud, self.negative_cloud), 1).to(self.device)
        num_batches = np.ceil(train_x.size(0) / batch_size).astype('int')
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.map_network.parameters()},
            {'params': self.alpha}
        ], learning_rate, weight_decay=weight_decay)
        training_loss = 0
        for i in range(num_epochs):
            for j in range(num_batches):
                if j < num_batches - 1:
                    x = train_x[j * batch_size: (j + 1) * batch_size]
                    y = self.partial_value_train[j * batch_size: (j + 1) * batch_size].to(self.device)
                    train_l = self.train_labels[j * batch_size: (j + 1) * batch_size].to(self.device)
                else:
                    x = train_x[(num_batches - 1) * batch_size:]
                    y = self.partial_value_train[(num_batches - 1) * batch_size:].to(self.device)
                    train_l = self.train_labels[(num_batches - 1) * batch_size:].to(self.device)
                optimizer.zero_grad()
                if kind == 'z':
                    output = self.get_posterior_zero(x, y, train_l)
                    loss = torch.mean(output)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()
                if kind == 'b':
                    output = self.get_posterior_binary(x, y, train_l)
                    # print(output)
                    loss = torch.mean(output)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()
            training_loss /= num_batches
            if i % print_every == 0:
                print(f"Epoch:{i}, Loss: {training_loss}")

    def get_posterior_test(self, points_to_predict=None):
        # check or create points to predict on
        if points_to_predict is None:
            points_to_predict = self.create_grid()
        # combine all data
        test_x = torch.cat((points_to_predict.repeat(self.test_partial.size(0), 1, 1), self.test_partial), 1).to(
            self.device)
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
        # set the covariance network to evaluation
        self.map_network.eval()
        # collect batch size
        bs = self.test_partial.size(0)
        for i in range(bs):
            # print(f'cloud {i}')
            mapped_cloud = self.map_norm(self.map_network(test_x[i]))
            cov_matrix_conditioned = self.covar_module_conditioned(
                self.enc_norm(test_x[i])).evaluate_kernel().to_dense().to(
                self.device)
            cov_matrix_mapping = self.covar_after_mapping(mapped_cloud).evaluate_kernel().to_dense().to(self.device)
            cov_matrix = self.alpha * cov_matrix_conditioned + (1 - self.alpha) * cov_matrix_mapping
            # cov_matrix = self.covar_after_mapping(mapped_cloud).evaluate_kernel().to_dense().to(self.device)
            print(cov_matrix)
            kernel_ff = cov_matrix[:points_to_predict.size(0), :points_to_predict.size(0)]
            kernel_pf = cov_matrix[points_to_predict.size(0):, :points_to_predict.size(0)]
            kernel_pp = cov_matrix[points_to_predict.size(0):, points_to_predict.size(0):]
            additional_noise = self.noise_var * torch.eye(self.partial_cloud.shape[1]).to(self.device)
            kernel_with_noise = (kernel_pp + additional_noise).to(self.device)
            posterior_mean = kernel_pf.T @ torch.linalg.inv(kernel_with_noise) @ self.partial_value_test[i].to(
                self.device)
            print(posterior_mean)
            posterior_var = kernel_ff - kernel_pf.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf
            posterior_diag = torch.diagonal(posterior_var, 0)
            print(posterior_var, posterior_diag)

    def predict(self, points_to_predict=None):
        # check or create points to predict on
        if points_to_predict is None:
            points_to_predict = self.create_grid()
        # combine all data
        test_x = torch.cat((points_to_predict.repeat(self.test_partial.size(0), 1, 1), self.test_partial), 1).to(
            self.device)
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
        # set the covariance network to evaluation
        self.map_network.eval()
        # collect batch size
        bs = self.test_partial.size(0)
        for i in range(bs):
            # print(f'cloud {i}')
            mapped_cloud = self.map_norm(self.map_network(test_x[i]))
            cov_matrix_conditioned = self.covar_module_conditioned(
                self.enc_norm(test_x[i])).evaluate_kernel().to_dense().to(
                self.device)
            cov_matrix_mapping = self.covar_after_mapping(mapped_cloud).evaluate_kernel().to_dense().to(self.device)
            cov_matrix = self.alpha * cov_matrix_conditioned + (1 - self.alpha) * cov_matrix_mapping
            # cov_matrix = self.covar_after_mapping(mapped_cloud).evaluate_kernel().to_dense().to(self.device)
            print(cov_matrix)
            kernel_ff = cov_matrix[:points_to_predict.size(0), :points_to_predict.size(0)]
            kernel_pf = cov_matrix[points_to_predict.size(0):, :points_to_predict.size(0)]
            kernel_pp = cov_matrix[points_to_predict.size(0):, points_to_predict.size(0):]
            additional_noise = self.noise_var * torch.eye(self.partial_cloud.shape[1]).to(self.device)
            kernel_with_noise = (kernel_pp + additional_noise).to(self.device)
            posterior_mean = kernel_pf.T @ torch.linalg.inv(kernel_with_noise) @ self.partial_value_test[i].to(
                self.device)
            print(posterior_mean)
            posterior_var = kernel_ff - kernel_pf.T @ torch.linalg.inv(kernel_with_noise) @ kernel_pf
            posterior_diag = torch.diagonal(posterior_var, 0)

            with torch.no_grad():
                prob_on_surface = norm.pdf(np.zeros(posterior_mean.shape), loc=posterior_mean.cpu().detach().numpy(),
                                           scale=np.sqrt(posterior_diag.cpu().detach().numpy()))
            gp = points_to_predict.cpu().numpy()
            gp_x = gp[:, 0].reshape(self.grid_sizes)
            gp_y = gp[:, 1].reshape(self.grid_sizes)
            gp_prob = prob_on_surface.reshape(self.grid_sizes)

            fig = plt.figure(figsize=(11, 4))
            ax = fig.add_subplot(111)
            plot = ax.pcolormesh(gp_x, gp_y, gp_prob, shading='gouraud', cmap='Greys')
            ax.scatter(self.test_partial[i].cpu().numpy()[:, 0], self.test_partial[i].cpu().numpy()[:, 1], c='r', s=1)
            fig.colorbar(plot)
            ax.axis('equal')
            ax.set_title(f'Probability of being on the surface')

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

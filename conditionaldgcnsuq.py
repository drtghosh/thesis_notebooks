import torch
import torch.nn as nn
import gpytorch
import numpy as np
from models import DeepGCN, PointNetEncoder


def compute_log_likelihood(data, label, mean, variance):
    dist_term = -0.5 * (label - mean).T @ torch.linalg.inv(variance) @ (label - mean)
    det_term = -0.5 * torch.log(torch.linalg.det(variance))
    const_term = -0.5 * data.size(0) * torch.log(2 * torch.tensor(torch.pi))
    log_likelihood = dist_term + det_term + const_term
    return log_likelihood


class ConditionalDGCN:
    """
    Class of Deep Gaussian Covariance Network intended to learn non-stationary hyperparameters
    of a Gaussian Process using Deep Learning for a given labeled dataset conditioned on direction
    (non-isotropic) and encoding of partially observed data/ points
    """

    def __init__(self, partial_data=None, remaining_data=None, test_data=None, training_label=None, test_label=None,
                 predict_noise=False, num_kernels=1, noise_variance=0.01, latent_dim=64):
        assert partial_data is not None
        assert remaining_data is not None
        assert test_data is not None
        assert training_label is not None
        assert test_label is not None
        self.partial_data = partial_data
        self.remaining_data = remaining_data
        self.test_data = test_data
        self.space_dim = partial_data.size(-1)
        self.num_kernels = num_kernels
        self.training_label = training_label
        self.test_label = test_label
        self.noise_variance = noise_variance
        self.latent_dim = latent_dim

        # set device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # initialize the Deep GCN network
        self.kernel_net = DeepGCN(32, 2, self.space_dim, 3)
        self.kernel_net.to(self.device)

        # initialize the encoder network
        self.encoder = PointNetEncoder(self.partial_data.size(1), self.space_dim, 2, 64, 3, 2, self.latent_dim)
        self.encoder.to(self.device)

    def create_conditional_data(self, k=5):
        training_partial = self.partial_data.unsqueeze(2).expand(-1, -1, k, -1)
        knn_dist_partial = torch.empty(training_partial.size())
        for i in range(self.partial_data.size(0)):
            points = self.partial_data[i]
            for j, p in enumerate(points):
                dist = points.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(5, largest=False, sorted=False)[1]
                knn_points = points.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_partial[i][j] = knn_points - p
        directed_partial = torch.concat((training_partial, knn_dist_partial), dim=3)
        directed_partial = directed_partial.reshape(directed_partial.size(0),
                                                    directed_partial.size(1) * directed_partial.size(2),
                                                    directed_partial.size(-1))

        training_remaining = self.remaining_data.unsqueeze(2).expand(-1, -1, k, -1)
        knn_dist_remaining = torch.empty(training_remaining.size())
        for i in range(self.remaining_data.size(0)):
            neighbors = self.partial_data[i]
            points = self.remaining_data[i]
            for j, p in enumerate(points):
                dist = neighbors.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(5, largest=False, sorted=False)[1]
                knn_points = neighbors.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_remaining[i][j] = knn_points - p
        directed_remaining = torch.concat((training_remaining, knn_dist_remaining), dim=3)
        directed_remaining = directed_remaining.reshape(directed_remaining.size(0),
                                                        directed_remaining.size(1) * directed_remaining.size(2),
                                                        directed_remaining.size(-1))

        labels_partial = self.noise_variance * torch.randn(self.partial_data.size()[:-1])
        labels_partial = labels_partial.unsqueeze(2).repeat(1, 1, k)
        labels_partial = labels_partial.reshape(labels_partial.size(0), -1)

        return directed_partial, directed_remaining, labels_partial

    def compute_kernel(self, x):
        scale_matrix = self.kernel_net(x)
        scaled_data = scale_matrix * x[..., None]
        reshaped_data = torch.permute(scaled_data, (2, 0, 1))
        kernel_matrix = torch.exp(-0.5 * torch.cdist(reshaped_data, reshaped_data)).mean(dim=0)
        return kernel_matrix, reshaped_data

    def compute_log_marginal(self, x, y):
        covar_matrix, reshaped_data = self.compute_kernel(x)
        noise = self.noise_variance * torch.eye(x.size(0)).to(self.device)
        covar_with_noise = covar_matrix + noise
        log_likelihood = compute_log_likelihood(x, y, 0, covar_with_noise)
        return log_likelihood

    def compute_posterior_batch(self, train_x, train_y, test_x, test_y, evaluate=False):
        covar_matrix_train, reshaped_data_train = self.compute_kernel(train_x)
        noise = self.noise_variance * torch.eye(train_x.size(0)).to(self.device)
        covar_with_noise = covar_matrix_train + noise
        if evaluate:
            self.kernel_net.eval()
        scale_matrix_test = self.kernel_net(test_x)
        scaled_data_test = scale_matrix_test * test_x[..., None]
        reshaped_data_test = torch.permute(scaled_data_test, (2, 0, 1))
        covar_matrix_test = torch.exp(-0.5 * torch.cdist(reshaped_data_test, reshaped_data_test)).mean(dim=0)
        covar_matrix_mixed = torch.exp(-0.5 * torch.cdist(reshaped_data_test, reshaped_data_train)).mean(dim=0)
        covar_inv = torch.linalg.inv(covar_with_noise)
        posterior_mean = covar_matrix_mixed @ covar_inv @ train_y
        posterior_var = covar_matrix_test - covar_matrix_mixed @ covar_inv @ covar_matrix_mixed.T
        # log_likelihood = compute_log_likelihood(test_x, test_y, posterior_mean, posterior_var)
        # return log_likelihood
        return posterior_mean, posterior_var

    def train_kernel(self, num_epochs=20, batch_size=20, learning_rate=0.001, weight_decay=1e-5, print_every=2, k=5):
        directed_partial, directed_remaining, labels_partial = self.create_conditional_data()
        partial = self.partial_data.to(self.device)
        # compute encoding
        encoding = self.encoder(partial.transpose(1, 2))



import torch
import torch.nn as nn
import gpytorch
import numpy as np
from models import DeepGCN, PointNetEncoder


class ConditionalDGCN:
    """
    Class of Deep Gaussian Covariance Network intended to learn non-stationary hyperparameters
    of a Gaussian Process using Deep Learning for a given labeled dataset conditioned on direction
    (non-isotropic) and encoding of partially observed data/ points
    """

    def __init__(self, training_data=None, test_data=None, training_label=None, test_label=None, predict_noise=False,
                 num_kernels=1, noise_variance=0.01):
        assert training_data is not None
        assert test_data is not None
        assert training_label is not None
        assert test_label is not None
        self.training_data = training_data
        self.test_data = test_data
        self.space_dim = training_data.size(1)
        self.num_kernels = num_kernels
        self.training_label = training_label
        self.test_label = test_label
        self.noise_variance = noise_variance

        # set device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # initialize the Deep GCN network
        self.kernel_net = DeepGCN(32, 2, self.space_dim, 2)
        self.kernel_net.to(self.device)

    def compute_kernel(self, x, y):
        scale_matrix = self.kernel_net(x)
        scaled_data = scale_matrix * x[..., None]
        kernel_matrix = torch.exp(-0.5 * torch.cdist(scaled_data, scaled_data)).mean(dim=0)

        return kernel_matrix, scaled_data

    def compute_log_marginal(self, x, y):
        covar_matrix, scaled_data = self.compute_kernel(x, y)
        noise = self.noise_variance * torch.eye(x.size(0)).to(self.device)
        covar_with_noise = covar_matrix + noise
        dist_term = -0.5 * y.T @ torch.linalg.inv(covar_with_noise) @ y
        det_term = -0.5 * torch.log(torch.linalg.det(covar_with_noise))
        const_term = -0.5 * x.size(0) * torch.log(2 * torch.Tensor(torch.pi))
        log_likelihood = dist_term + det_term + const_term
        return log_likelihood

    def compute_posterior_batch(self, train_x, train_y, test_x, test_y, evaluate=False):
        covar_matrix_train, scaled_data_train = self.compute_kernel(train_x, train_y)
        noise = self.noise_variance * torch.eye(train_x.size(0)).to(self.device)
        covar_with_noise = covar_matrix_train + noise
        if evaluate:
            self.kernel_net.eval()
        scale_matrix_test = self.kernel_net(test_x)
        scaled_data_test = scale_matrix_test * test_x[..., None]
        covar_matrix_test = torch.exp(-0.5 * torch.cdist(scaled_data_test, scaled_data_test)).mean(dim=0)
        covar_matrix_mixed = torch.exp(-0.5 * torch.cdist(scaled_data_test, scaled_data_train)).mean(dim=0)
        covar_inv = torch.linalg.inv(covar_with_noise)
        posterior_mean = covar_matrix_mixed @ covar_inv @ train_y
        posterior_var = covar_matrix_test - covar_matrix_mixed @ covar_inv @ covar_matrix_mixed.T
        dist_term = -0.5 * (test_y - posterior_mean).T @ torch.linalg.inv(posterior_var) @ (test_y - posterior_mean)
        det_term = -0.5 * torch.log(torch.linalg.det(posterior_var))
        const_term = -0.5 * test_x.size(0) * torch.log(2 * torch.Tensor(torch.pi))
        log_likelihood = dist_term + det_term + const_term
        return log_likelihood

    def train_kernel(self, num_epochs=20, batch_size=20, learning_rate=0.001, weight_decay=1e-5, use_posterior=False):
        train_x = self.training_data.to(self.device)
        train_y = self.training_label.to(self.device)
        num_batches = np.ceil(train_x.size(0) / batch_size).astype('int')
        optimizer = torch.optim.AdamW([
            {'params': self.kernel_net.parameters()},
        ], learning_rate, weight_decay=weight_decay)
        training_loss = 0
        for i in range(num_epochs):
            for j in range(num_batches):
                if j < num_batches - 1:
                    x = train_x[j * batch_size: (j + 1) * batch_size]
                    y = train_y[j * batch_size: (j + 1) * batch_size]
                else:
                    x = train_x[(num_batches - 1) * batch_size:]
                    y = train_y[(num_batches - 1) * batch_size:]
                optimizer.zero_grad()


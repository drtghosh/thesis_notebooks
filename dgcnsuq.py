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


class DGCN:
    """
    Class of Deep Gaussian Covariance Network intended to learn non-stationary hyperparameters
    of a Gaussian Process using Deep Learning for a given labeled dataset
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
        self.kernel_net = DeepGCN(32, 2, self.space_dim, 3)
        self.kernel_net.to(self.device)

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

    def train_kernel(self, num_epochs=20, batch_size=20, learning_rate=0.001, weight_decay=1e-5, print_every=2,
                     use_posterior=False):
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
                if use_posterior:
                    output = self.compute_posterior_batch(x, y, x, y)
                    print(output)
                else:
                    output = -self.compute_log_marginal(x, y)
                    loss = torch.mean(output)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()
            training_loss /= num_batches
            if i % print_every == 0:
                print(f"Epoch:{i}, Loss: {training_loss}")

import torch
import torch.nn as nn
import gpytorch
import numpy as np
from models import DeepGCN, PointNetEncoder


def compute_log_likelihood(data, label, mean, variance):
    dist_term = -0.5 * (label - mean).T @ torch.linalg.inv(variance) @ (label - mean)
    det_term = -0.5 * torch.log(torch.linalg.det(variance) + 1e-6)
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
                 predict_noise=False, num_kernels=1, noise_variance=0.01, latent_dim=64, grid_size=100, k=5):
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
        self.grid_sizes = np.ones(self.space_dim, dtype=np.int32) * grid_size
        self.k = k

        # set device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # initialize the Deep GCN network
        self.kernel_net = DeepGCN(32, 2, self.space_dim, 2 * self.space_dim + latent_dim, 3)
        self.kernel_net.to(self.device)

        # initialize the encoder network
        self.encoder = PointNetEncoder(self.partial_data.size(1), self.space_dim, 2, 64, 3, 2, self.latent_dim)
        self.encoder.to(self.device)

    def create_conditional_data_training(self):
        training_partial = self.partial_data.unsqueeze(2).expand(-1, -1, self.k, -1)
        knn_dist_partial = torch.empty(training_partial.size())
        for i in range(self.partial_data.size(0)):
            points = self.partial_data[i]
            for j, p in enumerate(points):
                dist = points.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(self.k, largest=False, sorted=False)[1]
                knn_points = points.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_partial[i][j] = knn_points - p
        directed_partial = torch.concat((training_partial, knn_dist_partial), dim=3)
        directed_partial = directed_partial.reshape(directed_partial.size(0),
                                                    directed_partial.size(1) * directed_partial.size(2),
                                                    directed_partial.size(-1))

        training_remaining = self.remaining_data.unsqueeze(2).expand(-1, -1, self.k, -1)
        knn_dist_remaining = torch.empty(training_remaining.size())
        for i in range(self.remaining_data.size(0)):
            neighbors = self.partial_data[i]
            points = self.remaining_data[i]
            for j, p in enumerate(points):
                dist = neighbors.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(self.k, largest=False, sorted=False)[1]
                knn_points = neighbors.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_remaining[i][j] = knn_points - p
        directed_remaining = torch.concat((training_remaining, knn_dist_remaining), dim=3)
        directed_remaining = directed_remaining.reshape(directed_remaining.size(0),
                                                        directed_remaining.size(1) * directed_remaining.size(2),
                                                        directed_remaining.size(-1))

        labels_partial = self.noise_variance * torch.randn(self.partial_data.size()[:-1])
        # labels_partial = labels_partial.un squeeze(2).repeat(1, 1, self.k)
        # labels_partial = labels_partial.reshape(labels_partial.size(0), -1)

        return directed_partial, directed_remaining, labels_partial

    def create_conditional_data_test(self):
        test_partial = self.test_data.unsqueeze(2).expand(-1, -1, self.k, -1)
        knn_dist_partial = torch.empty(test_partial.size())
        for i in range(self.test_data.size(0)):
            points = self.test_data[i]
            for j, p in enumerate(points):
                dist = points.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(self.k, largest=False, sorted=False)[1]
                knn_points = points.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_partial[i][j] = knn_points - p
        directed_partial = torch.concat((test_partial, knn_dist_partial), dim=3)
        directed_partial = directed_partial.reshape(directed_partial.size(0),
                                                    directed_partial.size(1) * directed_partial.size(2),
                                                    directed_partial.size(-1))
        test_grid = self.create_grid()
        test_grid_repeated = test_grid.repeat(self.test_data.size(0), 1, 1).to(self.device)
        test_full = test_grid_repeated.unsqueeze(2).expand(-1, -1, self.k, -1)
        knn_dist_full = torch.empty(test_full.size())
        for i in range(self.test_data.size(0)):
            neighbors = self.test_data[i]
            points = test_grid_repeated[i]
            for j, p in enumerate(points):
                dist = neighbors.add(-p).pow(2).sum(dim=1)
                knn_indices = dist.topk(self.k, largest=False, sorted=False)[1]
                knn_points = neighbors.gather(0, knn_indices.unsqueeze(-1).repeat(1, 2))
                knn_dist_full[i][j] = knn_points - p
        directed_full = torch.concat((test_grid_repeated, knn_dist_full), dim=3)
        directed_full = directed_full.reshape(directed_full.size(0), directed_full.size(1) * directed_full.size(2),
                                              directed_full.size(-1))

        return test_grid_repeated, directed_partial, directed_full

    def compute_kernel(self, conditioned_x, x, k):
        scale_matrix_k = self.kernel_net(conditioned_x)
        scale_matrix = scale_matrix_k.reshape(x.size(0), k, scale_matrix_k.size(1), scale_matrix_k.size(2)).mean(dim=1)
        scaled_data = scale_matrix * x[..., None]
        reshaped_data = torch.permute(scaled_data, (2, 0, 1))
        kernel_matrix = torch.exp(-0.5 * torch.cdist(reshaped_data, reshaped_data)).mean(dim=0)
        return kernel_matrix, reshaped_data

    def compute_log_marginal(self, conditioned_x, x, y, k):
        covar_matrix, reshaped_data = self.compute_kernel(conditioned_x, x, k)
        noise = self.noise_variance * torch.eye(x.size(0)).to(self.device)
        covar_with_noise = covar_matrix + noise
        log_likelihood = compute_log_likelihood(x, y, 0, covar_with_noise)
        return log_likelihood

    def compute_posterior(self, k=5):
        directed_partial, directed_remaining, labels_partial = self.create_conditional_data_training()
        partial = self.partial_data.to(self.device)
        remaining = self.remaining_data.to(self.device)
        # compute encoding
        encoding = self.encoder(partial.transpose(1, 2))
        # repeat encoding for each point in partial data
        directed_partial = torch.cat((directed_partial, encoding.unsqueeze(1).repeat(1, directed_partial.size(1), 1)),
                                     2)
        directed_remaining = torch.cat(
            (directed_remaining, encoding.unsqueeze(1).repeat(1, directed_remaining.size(1), 1)),
            2)
        bs = partial.size(0)
        posterior_loss = torch.empty(bs).to(self.device)
        for i in range(bs):
            partial_x = partial[i]
            remaining_x = remaining[i]
            partial_y = labels_partial[i]
            conditioned_partial_x = directed_partial[i]
            conditioned_remaining_x = directed_remaining[i]
            covar_matrix_partial, reshaped_data_partial = self.compute_kernel(conditioned_partial_x, partial_x, k)
            covar_matrix_remaining, reshaped_data_remaining = self.compute_kernel(conditioned_remaining_x, remaining_x,
                                                                                  k)
            noise = self.noise_variance * torch.eye(covar_matrix_partial.size(0)).to(self.device)
            covar_with_noise = covar_matrix_partial + noise
            covar_matrix_pr = torch.exp(-0.5 * torch.cdist(reshaped_data_remaining, reshaped_data_partial)).mean(dim=0)
            covar_inverse = torch.linalg.inv(covar_with_noise)
            posterior_mean = covar_matrix_pr @ covar_inverse @ partial_y
            posterior_var = covar_matrix_remaining - covar_matrix_pr @ covar_inverse @ covar_matrix_pr.T
            log_likelihood = compute_log_likelihood(remaining_x, 0, posterior_mean, posterior_var)
            posterior_loss[i] = -log_likelihood
        return posterior_loss

    def train_kernel(self, num_epochs=20, learning_rate=0.001, weight_decay=1e-5, print_every=2, k=5):
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.kernel_net.parameters()}
        ], learning_rate, weight_decay=weight_decay)
        for n in range(num_epochs):
            output = self.compute_posterior(k=k)
            loss = torch.mean(output)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            if n % print_every == 0:
                print(f"Epoch:{n}, Loss: {training_loss}")

    def predict(self):
        test_grid_repeated, directed_partial, directed_full = self.create_conditional_data_test()
        test = self.test_data.to(self.device)
        full = test_grid_repeated.to(self.device)
        # compute encoding
        encoding = self.encoder(test.transpose(1, 2))
        # repeat encoding for each point in partial data
        directed_partial = torch.cat((directed_partial, encoding.unsqueeze(1).repeat(1, directed_partial.size(1), 1)),
                                     2)
        directed_full = torch.cat(
            (directed_full, encoding.unsqueeze(1).repeat(1, directed_full.size(1), 1)),
            2)
        bs = test.size(0)
        

    def create_grid(self, box_min=None, box_max=None, eps=0.1):
        # find the bounding box for all dataset
        if box_min is None:
            box_min = torch.amin(self.test_data, (1, 0)) - eps
        if box_max is None:
            box_max = torch.amax(self.test_data, (1, 0)) + eps

        # Build a grid (dimension-agnostic)
        grid_vertices = np.meshgrid(
            *[np.linspace(box_min[d], box_max[d], self.grid_sizes[d]) for d in range(self.space_dim)])
        grid_vertices = np.stack(grid_vertices, axis=-1).reshape(-1, self.space_dim)
        grid_vertices = torch.tensor(grid_vertices, dtype=torch.float32)
        return grid_vertices

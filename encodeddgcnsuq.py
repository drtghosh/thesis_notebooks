import torch
# import torch.nn as nn
# import gpytorch
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch.nn.functional as fn
from models import DeepGCN, PointNetEncoder


def compute_log_likelihood(data, label, mean, variance):
    dist_term = -0.5 * (label - mean).T @ torch.linalg.inv(variance) @ (label - mean)
    det_term = -0.5 * torch.log(torch.linalg.det(variance) + 1e-6)
    # const_term = -0.5 * data.size(0) * torch.log(2 * torch.tensor(torch.pi))
    log_likelihood = dist_term + det_term  # + const_term
    return log_likelihood


class EncodedDGCN:
    """
    Class of Deep Gaussian Covariance Network intended to learn non-stationary hyperparameters
    of a Gaussian Process using Deep Learning for a given labeled dataset conditioned on direction
    (non-isotropic) and encoding of partially observed data/ points
    """

    def __init__(self, partial_data=None, remaining_data_pos=None, remaining_data_neg=None, test_data=None,
                 training_label=None, test_label=None, predict_noise=False, num_kernels=1, noise_variance=0.01,
                 latent_dim=64, grid_size=100):
        assert partial_data is not None
        assert remaining_data_pos is not None
        assert remaining_data_neg is not None
        assert test_data is not None
        assert training_label is not None
        assert test_label is not None
        self.partial_data = partial_data
        self.remaining_data_pos = remaining_data_pos
        self.remaining_data_neg = remaining_data_neg
        self.test_data = test_data
        self.space_dim = partial_data.size(-1)
        self.num_kernels = num_kernels
        self.training_label = training_label
        self.test_label = test_label
        self.noise_variance = noise_variance
        self.latent_dim = latent_dim
        self.grid_sizes = np.ones(self.space_dim, dtype=np.int32) * grid_size
        self.labels_partial = self.noise_variance * torch.randn(self.partial_data.size()[:-1])
        self.labels_test = self.noise_variance * torch.randn(self.test_data.size()[:-1])

        # set device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # initialize the Deep GCN network
        self.kernel_net = DeepGCN(32, 2, self.space_dim, self.space_dim + latent_dim, num_kernels)
        self.kernel_net.to(self.device)

        # initialize the encoder network
        self.encoder = PointNetEncoder(self.partial_data.size(1), self.space_dim, 2, 64, 3, 2, self.latent_dim)
        self.encoder.to(self.device)

    def compute_kernel(self, conditioned_x, x):
        scale_matrix = self.kernel_net(conditioned_x)
        scaled_data = scale_matrix * x[..., None]
        reshaped_data = torch.permute(scaled_data, (2, 0, 1))
        kernel_matrix = torch.exp(-0.5 * torch.cdist(reshaped_data, reshaped_data)).mean(dim=0)
        return kernel_matrix, reshaped_data

    def compute_log_marginal(self, conditioned_x, x, y):
        covar_matrix, reshaped_data = self.compute_kernel(conditioned_x, x)
        noise = self.noise_variance * torch.eye(x.size(0)).to(self.device)
        covar_with_noise = covar_matrix + noise
        log_likelihood = compute_log_likelihood(x, y, 0, covar_with_noise)
        return log_likelihood

    def compute_posterior(self):
        partial = self.partial_data.to(self.device)
        labels = self.labels_partial.to(self.device)
        remaining_pos = self.remaining_data_pos.to(self.device)
        remaining_neg = self.remaining_data_neg.to(self.device)
        # compute encoding
        encoding = self.encoder(partial.transpose(1, 2))
        # repeat encoding for each point in partial data
        encoded_partial = torch.cat((partial, encoding.unsqueeze(1).repeat(1, partial.size(1), 1)), 2)
        encoded_remaining_pos = torch.cat((remaining_pos, encoding.unsqueeze(1).repeat(1, remaining_pos.size(1), 1)), 2)
        encoded_remaining_neg = torch.cat((remaining_neg, encoding.unsqueeze(1).repeat(1, remaining_neg.size(1), 1)), 2)
        bs = partial.size(0)
        posterior_loss = torch.empty(bs).to(self.device)
        for i in range(bs):
            partial_x = partial[i]
            remaining_pos_x = remaining_pos[i]
            remaining_neg_x = remaining_neg[i]
            partial_y = labels[i]
            conditioned_partial_x = encoded_partial[i]
            conditioned_remaining_pos_x = encoded_remaining_pos[i]
            conditioned_remaining_neg_x = encoded_remaining_neg[i]
            covar_matrix_partial, reshaped_data_partial = self.compute_kernel(conditioned_partial_x, partial_x)
            covar_matrix_remaining_pos, reshaped_data_remaining_pos = self.compute_kernel(conditioned_remaining_pos_x,
                                                                                          remaining_pos_x)
            covar_matrix_remaining_neg, reshaped_data_remaining_neg = self.compute_kernel(conditioned_remaining_neg_x,
                                                                                          remaining_neg_x)
            noise = self.noise_variance * torch.eye(covar_matrix_partial.size(0)).to(self.device)
            covar_with_noise = covar_matrix_partial + noise
            covar_matrix_rp_pos = torch.exp(
                -0.5 * torch.cdist(reshaped_data_remaining_pos, reshaped_data_partial)).mean(dim=0)
            covar_matrix_rp_neg = torch.exp(
                -0.5 * torch.cdist(reshaped_data_remaining_neg, reshaped_data_partial)).mean(dim=0)
            covar_inverse = torch.linalg.inv(covar_with_noise)
            posterior_mean_pos = covar_matrix_rp_pos @ covar_inverse @ partial_y
            posterior_var_pos = covar_matrix_remaining_pos - covar_matrix_rp_pos @ covar_inverse @ covar_matrix_rp_pos.T
            log_likelihood_pos = compute_log_likelihood(remaining_pos_x, 0, posterior_mean_pos, posterior_var_pos)
            posterior_mean_neg = covar_matrix_rp_neg @ covar_inverse @ partial_y
            posterior_var_neg = covar_matrix_remaining_neg - covar_matrix_rp_neg @ covar_inverse @ covar_matrix_rp_neg.T
            log_likelihood_neg = compute_log_likelihood(remaining_neg_x, 0, posterior_mean_neg, posterior_var_neg)
            loss_entropy = - (
                (fn.softmax(covar_matrix_partial, dim=1) * fn.log_softmax(covar_matrix_partial, dim=1)).sum(
                    dim=1)).mean()
            posterior_loss[i] = -log_likelihood_pos + log_likelihood_neg + loss_entropy
        return posterior_loss

    def train_kernel(self, num_epochs=20, learning_rate=0.001, weight_decay=1e-5, print_every=2):
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.kernel_net.parameters()}
        ], learning_rate, weight_decay=weight_decay)
        for n in range(num_epochs):
            output = self.compute_posterior()
            loss = torch.mean(output)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            if n % print_every == 0:
                print(f"Epoch:{n}, Loss: {training_loss}")

    def predict(self):
        test = self.test_data.to(self.device)
        labels_test = self.labels_test.to(self.device)
        test_grid = self.create_grid()
        test_grid_repeated = test_grid.repeat(self.test_data.size(0), 1, 1).to(self.device)
        full = test_grid_repeated.to(self.device)
        # compute encoding
        encoding = self.encoder(test.transpose(1, 2))
        # repeat encoding for each point in partial data
        encoded_partial = torch.cat((test, encoding.unsqueeze(1).repeat(1, test.size(1), 1)), 2)
        encoded_full = torch.cat(
            (test_grid_repeated, encoding.unsqueeze(1).repeat(1, test_grid_repeated.size(1), 1)),
            2)
        bs = test.size(0)
        for i in range(bs):
            test_x = test[i]
            full_x = full[i]
            test_y = labels_test[i]
            conditioned_test_x = encoded_partial[i]
            conditioned_full_x = encoded_full[i]
            covar_matrix_test, reshaped_data_test = self.compute_kernel(conditioned_test_x, test_x)
            covar_matrix_full, reshaped_data_full = self.compute_kernel(conditioned_full_x, full_x)
            noise = self.noise_variance * torch.eye(covar_matrix_test.size(0)).to(self.device)
            covar_with_noise = covar_matrix_test + noise
            covar_matrix_tf = torch.exp(-0.5 * torch.cdist(reshaped_data_full, reshaped_data_test)).mean(dim=0)
            covar_inverse = torch.linalg.inv(covar_with_noise)
            posterior_mean = covar_matrix_tf @ covar_inverse @ test_y
            print(covar_matrix_test)
            posterior_var = covar_matrix_full - covar_matrix_tf @ covar_inverse @ covar_matrix_tf.T
            posterior_diag = torch.diagonal(posterior_var, 0)

            with torch.no_grad():
                prob_on_surface = norm.pdf(np.zeros(posterior_mean.shape), loc=posterior_mean.cpu().detach().numpy(),
                                           scale=np.sqrt(posterior_diag.cpu().detach().numpy()))
            gp = full_x.cpu().numpy()
            gp_x = gp[:, 0].reshape(self.grid_sizes)
            gp_y = gp[:, 1].reshape(self.grid_sizes)
            gp_prob = prob_on_surface.reshape(self.grid_sizes)

            fig = plt.figure(figsize=(11, 4))
            ax = fig.add_subplot(111)
            plot = ax.pcolormesh(gp_x, gp_y, gp_prob, shading='gouraud', cmap='Greys')
            ax.scatter(test_x.cpu().numpy()[:, 0], test_x.cpu().numpy()[:, 1], c='b', s=0.1)
            fig.colorbar(plot)
            ax.axis('equal')
            ax.set_title(f'Probability of being on the surface')

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

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
                 num_kernels=1):
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

        # set device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # initialize the Deep GCN network
        self.kernel_net = DeepGCN(32, 2, self.space_dim, 2)

    def compute_marginal(self, x, y):
        scale_matrix = self.kernel_net(x)

    def train_kernel(self, num_epochs=20, batch_size=20, learning_rate=0.001, weight_decay=1e-5):
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


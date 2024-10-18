import torch
import torch.nn as nn
import gpytorch
import numpy as np
from models import MLPGrow, PointNetEncoder


class ConditionalDGCN:
    """
    Class of Deep Gaussian Covariance Network intended to learn non-stationary hyperparameters
    of a Gaussian Process using Deep Learning for a given labeled dataset conditioned on direction
    (non-isotropic) and encoding of partially observed data/ points
    """
    def __init__(self, training_data=None, test_data=None, predict_noise=False, num_kernels=1):
        assert training_data is not None
        assert test_data is not None
        self.training_data = training_data
        self.test_data = test_data
        self.space_dim = training_data.size(1)
        self.num_kernels = num_kernels

        
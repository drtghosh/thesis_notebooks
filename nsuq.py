import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox/src')))
import platform
if platform.processor()=='arm':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox/build-arm')))
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox/build')))
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/torch-mvnorm/')))
from mvnorm import multivariate_normal_cdf as Phi, integration
import gpytoolbox
from .models import MLP, covariance_network, deep_ritz_network
from .misc import project_to_psd_torch
from .datasets import grad_covariance_dataset, grad_mean_dataset, latent_dataset
from itertools import cycle
import matplotlib.pyplot as plt
import pickle
import scipy
import multiprocessing
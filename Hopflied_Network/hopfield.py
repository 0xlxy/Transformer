#%% Imports
from PIL.Image import NONE
import numpy as np
import scipy.integrate as si
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
import statsmodels as stm
import scipy.stats as st
from importlib import reload
import TimeSeriesTransformer as TST

#%% Define Hopfield Network
def lorenz(x, t, sigma=10, rho=28, beta=8/3):
    y = np.zeros(3)
    y[0] = sigma * (x[1] - x[0])
    y[1] = x[0] * (rho - x[2]) - x[1]
    y[2] = x[0] * x[1] - beta * x[2]
    return y
    
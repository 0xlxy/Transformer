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
import seaborn as sns
from mpl_toolkits import mplot3d
import statsmodels as stm
import scipy.stats as st
from importlib import reload
import TimeSeriesTransformer as TST;

#%% Define Anti-hebbian System - extremely complicated!!
# very difficult to study! (generate compelx dynamic)

def antihebbian(y, t, alpha = 0.05, ndim = 5):
  x = y[:ndim] # activities of the neurons
  W = y[ndim:].reshape(ndim,-1) # couplings
  dydt = np.zeros(y.shape)
  dydt[:ndim] = np.dot(W, x)
  dydt[ndim:] = alpha * (np.identity(ndim) - np.outer(x,x)).reshape(-1)
  return dydt

ndim = 5 # number of units in the system
y0 = np.random.rand(ndim+ndim*ndim)

t = np.linspace(0,1000,1000000)
y = si.odeint(antihebbian, y0, t)

# train on x instead of y - y[:,:5]
# train on the activity space (first ndim)
# x_dot = -x + g(W_x)
# x_dot: vector of firing rate

#%% Plot Anti-hebbian System

plt.figure()
plt.clf()
ax = plt.subplot(2,2,1)
plt.plot(t, y[:,:ndim])
plt.subplot(2,2,2, sharex=ax)
plt.plot(t, y[:,ndim:])
plt.subplot(2,2,3)
plt.imshow(y[:,:ndim].T, aspect='auto')
plt.tight_layout()
plt.subplot(2,2,4)
plt.imshow(y[:,ndim:].T, aspect='auto')
plt.tight_layout()

#%%
# generate data 

def generate_data(y, tdim, odim, nsamples = 100000):
  i = np.random.choice(np.arange(len(y)-odim-tdim),size=nsamples,replace=False)
  # print('i shape', i.shape)
  X=y[[np.arange(ii,ii+tdim) for ii in i]]
  # print('X array shape', X.shape)
  Y=y[[np.arange(ii+tdim,ii+tdim+odim) for ii in i]]
  # print('Y array shape', Y.shape)
  X = torch.tensor(X).float()
  Y = torch.tensor(Y).float()
  Y = torch.reshape(Y,(nsamples,-1))
  return X,Y

print('y len', len(y))
X,Y = generate_data(y, 5, 1, 10000)
# X,Y = generate_data(x, 5, 1, 1);

print(X)
print("X shape", X.shape) # torch.Size([10000, 5, 30])
print(Y)
print("Y shape", Y.shape) # torch.Size([10000, 30])

#%% plot data
# fig = plt.figure() 
# ax = plt.axes(projection='3d')

# # Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

#%%
reload(TST)

ndim = 30
tdim = 5
odim = 1
kdim = 5
vdim = 5
fdim = 20
tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)

#loss = F.cross_entropy
loss = F.mse_loss

def train_batch(x, y, model, optimizer, loss = loss, iterations = 1000):
  for iteration in range(iterations):
    #print("Iteration %d / %d" % (epoch, epochs));
    opt.zero_grad()
    o = model(x)
    l = loss(o, y)
    l.backward()
    opt.step()

  return l.item()

def train(x, model, optimizer, loss = loss, epochs = 10000, iterations = 1000, nsamples=100):
# def train(x, model, optimizer, loss = loss, epochs = 1000, iterations = 1000, nsamples=100):
  avg_err = 0
  for epoch in range(epochs):
    X,Y = generate_data(x, tdim, odim, nsamples=nsamples)
    l = train_batch(X, Y, model, optimizer, loss=loss, iterations=iterations)
    print('Epoch %d/%d: error: %r' % (epoch, epochs, l))
    if epoch + 10 >= epochs:
      avg_err += l
  avg_err = avg_err/10
  print('Average Error: ', avg_err)
  
learning_rate = 0.1

opt = optim.SGD(tst.parameters(), lr=learning_rate) #define optimizer
opt = optim.Adam(tst.parameters(), lr=1e-3)


# %%

result = train(y, tst, opt, loss, epochs=50)

# %% Save Model
torch.save(tst.state_dict(), 'TST_anti-hebbian_trained 2021-10-22 10000-1000-100-r-v.pth')

#%% Load Model
tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
tst.load_state_dict(torch.load('TST_anti-hebbian_trained 2021-10-22 10000-1000-100-r-v.pth'))

tst.eval()

#%% Test Model and Print Error
X,Y = generate_data(y, 5, 1, 1000)
Yhat = tst(X)
# print("X shape", X.shape)
# print(X)
# print("Y shape", Y.shape)
# print(Y)
# print("Yhat shape", Yhat.shape)
# print(Yhat)
loss = F.mse_loss
l = loss(Yhat, Y)
print('Yhat', len(Yhat))
print('Y', len(Y))
# look into loss function - campare each element indivicually
# errors = [loss(Yhat[i], Y[i]).item() for i in range(len(Y))]
# errors = loss(Yhat, Y, reduction='none')
errors = np.sum(np.square(Y.detach().numpy() - Yhat.detach().numpy()), axis=-1)
# print('errors', errors.shape)
print('Test Set Error', l.item())

# Error plots:
# For plots with high dimensions:
# PCA (Principle Component Analysis)
# plot a line for each dimension
# the dynamic over time 
# one line for each plot - color the line with errors 
# plot 30 lines with some offset!

# Attention plots:
# 5 * 5
# take a certain time window

# randomly plot 3 dimensions - 10 plots 
# 10 plots for the entire 30-dim space - make sure the color code have the same scales
# 1) 30 lines
# 2) 10 plots
# 3) PCA - plot the 3 dimsions with the highest variance - highest dimensional space - highest error!
# plot the totol error - plot as lorenz system

#%% Plot Error

fig = plt.figure() 
fig.set_size_inches(40, 8)
plt.clf()
for d in range(ndim):
  plt.subplot(1,ndim,1+d)
  plt.scatter(Y[:,d].detach().numpy(),Yhat[:,d].detach().numpy())
  plt.plot([Y[:,d].min(), Y[:,d].max()],[Y[:,d].min(), Y[:,d].max()])
  plt.title('dim=%d' % d)

#%% Plot Error Histogram
plt.figure()
plt.clf()
plt.hist(errors, bins=32)
plt.show()

# %% Make Error Dot Plot

fig = plt.figure() 
fig.set_size_inches(10, 8)
plt.clf()
# ax = fig.add_subplot(projection='3d')
coords = [i[-1] for i in X] # 1000 * 30
coords = np.array([n.numpy() for n in coords])
means = np.mean(coords)
coords_centered = coords - means
cov = coords.T @ coords / len(coords)
u, s, vt = np.linalg.svd(coords, full_matrices = False)
s = np.diag(s)
# eigenvalues * len(X) - (30,)
# XTX_eigenvalues - (30,)
pc = coords @ vt.T

ax = sns.lineplot(x = np.arange(1, 6), y = s ** 2 / sum(s ** 2))
ax.set_xticks(np.arange(1, 6))
ax.set_xlabel('$i^{th}$ Principal Component')
ax.set_ylabel('Proportion of Variance')


# loss = F.mse_loss
# errors = [loss(Yhat[i], Y[i]).item() for i in range(len(Y))]

# p = ax.scatter(xs, ys, zs, c = errors, cmap = 'viridis')
errors_plot = errors.copy()
# errors_plot[errors_plot > .01] = .01
plt.scatter(xs, ys, c = errors_plot, cmap = 'plasma')

# https://matplotlib.org/stable/tutorials/colors/colormaps.html


plt.xlabel('x-axis')
plt.ylabel('y-axis')
cbaxes = fig.add_axes([.95, .2, .05, .6])  # horizontal shift, vertical shift, bar width, bar height
plt.colorbar(cax = cbaxes)

plt.show()

# subplot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
# https://stackoverflow.com/questions/28144142/how-can-i-generate-a-colormap-array-from-a-simple-array-in-matplotlib

# %% key-query plot prep
na = 2000
X,Y = generate_data(x, tdim, odim, nsamples=na)
a = tst.attention(X)
print(a.shape)

# ref = query
# leg = key

# %% key-query plot
fig = plt.figure(25) 
fig.set_size_inches(25, 20)

plt.clf()
for tref in range(tdim):
  for t in range(tdim):
    ax = fig.add_subplot(tdim, tdim, tdim*tref+t+1,)
    ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = a[:na,tref,t].detach().numpy(), cmap = 'plasma')
    plt.title("ref: %d  lag=%d" % (tdim-tref, tdim-t))
    cbaxes = fig.add_axes([.94, .15, .04, .7])
    p = ax.scatter(xs, ys, c = errors_plot, cmap = 'plasma')
    plt.colorbar(p, cax = cbaxes)

# 1) generate a new window and rescale it - open a new matplotlib window instead of in-line plotting - plot in separate window
# 2) try less plots (as long as get the structures)

# fig.savefig('../images/attention_plot_2000-10000.png', dpi=1000)


# one plot: x points in orignals sampe

# %% Transform Plot Prep
na = 1000
X,Y = generate_data(x, tdim, odim, nsamples=na)

X_arr = X.detach().numpy()[:,-1].copy()
for d in range(2):
  X_arr[:,d] -= X_arr[:,d].min()
  X_arr[:,d] *= 1/X_arr[:,d].max()

# %% K plot

fig = plt.figure()
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1)
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = X_arr)

K = tst.K(X) 
ax = fig.add_subplot(1, 2, 2)
plt.title('Transform to K')
ax.scatter(K[:na,-1,0].detach().numpy(), K[:na,-1,1].detach().numpy(), c = X_arr)
plt.show()

# %% Q plot

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy(), c = X_arr)

Q = tst.Q(X) 
ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.title('Transform to Q')
ax.scatter(Q[:na,-1,0].detach().numpy(), Q[:na,-1,1].detach().numpy(), Q[:na,-1,2].detach().numpy(), c = X_arr)
plt.show()

# %% V plot

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy(), c = X_arr)

V = tst.get_v(X).reshape(na, tdim, 3)
ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.title('Transform to V')
ax.scatter(V[:na,-1,0].detach().numpy(), V[:na,-1,1].detach().numpy(), V[:na,-1,2].detach().numpy(), c = X_arr)
plt.show()
# %% R plot

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1, projection='3d')
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy(), c = X_arr)

R = tst.get_r(X).reshape(na, tdim, 3)
ax = fig.add_subplot(1, 2, 2, projection='3d')
plt.title('Transform to R')
ax.scatter(R[:na,-1,0].detach().numpy(), R[:na,-1,1].detach().numpy(), R[:na,-1,2].detach().numpy(), c = X_arr)
plt.show()


# %% Timestep Evolution

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy())
ax.scatter(X[0,:,0].detach().numpy(), X[0,:,1].detach().numpy(), X[0,:,2].detach().numpy(), c = 'red', s = 200) # Color with red!
plt.show()

# %%
# Plotting the results as training
# Plot the errors while training
# Save the plots in a picture file - look at the pictures! , .save_fig
# put a date on the files
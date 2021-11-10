#%% Imports
from operator import mod
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

#%% Define Kuramoto Model
# live on a circle - 0-360

ndim = 5 # 5 couples oscillators
K = np.random.rand(ndim, ndim)
K[np.random.rand(ndim, ndim) <= 0.2] = 0
omega = 2 * np.random.rand(ndim) - 1

def kuramoto(phi,t):
  return omega + np.sum(K * np.sin(np.subtract.outer(phi,phi)), axis=0)
# sin (phi) = sin (phi + 2pi)
# RANDOM initial condition between 0 and 2pi

t = np.linspace(0,200,200*10)
phi_list = []
for _ in range(2**ndim):
  phi0 = 2 * np.pi * np.random.rand(ndim)
  phi = si.odeint(kuramoto, phi0, t)
  phi_mod = phi % (2* np.pi)
  phi_list.append(phi_mod)
phi_list = np.array(phi_list)

# shape is (10000, 5)

# the system is rotating
# Calculate difference on the circle - mod 360 / 2pi
# instead of run one long simulation --> take random starting point - simulate a bit

#%% Plot

plt.figure()
plt.clf()
plt.plot(t, phi_mod)
# time vs. lifted phase 

#%%

# generate data - random kuramoto system

def generate_data(x, tdim, odim, nsamples = 100):
  i = np.random.choice(np.arange(len(x)-odim-tdim),size=nsamples,replace=False)
  # print('i shape', i.shape)
  phi_mod = phi_list[np.random.randint(len(phi_list))] # choose a simulation by random -> sample from the simulation
  # approach2: from each data point, sample list ID --> take random data 
  X=phi_mod[[np.arange(ii,ii+tdim) for ii in i]]
  Y=phi_mod[[np.arange(ii+tdim,ii+tdim+odim) for ii in i]]
  X = torch.tensor(X).float()
  Y = torch.tensor(Y).float()
  Y = torch.reshape(Y,(nsamples,-1))
  return X,Y

print('x len', len(phi_mod))
X,Y = generate_data(phi_mod, 5, 1, 100)
# X,Y = generate_data(x, 5, 1, 1);

print(X)
print("X shape", X.shape) # torch.Size([100, 5, 3])
print(Y)
print("Y shape", Y.shape) # torch.Size([100, 3])

# 5 oscillator = 5 lines --> plot error as color for line
# take differecne between prediction and actually

# x: time y: phases (values)
# shift

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
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# %% Generate data & Plot
X,Y = generate_data(phi_mod, 5, 1, 1000)
print(X.shape)

plt.figure() 
plt.clf()
for i in range(X.shape[0]):
  plt.plot(X[i,:,0], X[i,:,1])
#%%
reload(TST)

ndim = 5
tdim = 5
odim = 1
kdim = 2
vdim = 2
fdim = 20
tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)

#loss = F.cross_entropy - write the own loss function (loss function on circular data)
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

result = train(phi_mod, tst, opt, loss, epochs=50)

# %% Save Model
torch.save(tst.state_dict(), 'TST_Kuramoto_trained 2021-10-22 v2.pth')

#%% Load Model
tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
# tst.load_state_dict(torch.load('TST_Lorenz_trained 2021-08-03 10000-1000-10000.pth'))
tst.load_state_dict(torch.load('TST_Kuramoto_trained 2021-10-22 v2.pth'))

tst.eval()

#%% Test Model and Print Error
X,Y = generate_data(phi_mod, 5, 1, 1000)
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

#%% Plot Error

plt.figure(11)
plt.clf()
for d in range(ndim):
  plt.subplot(1,ndim,1+d)
  plt.scatter(Y[:,d].detach().numpy(),Yhat[:,d].detach().numpy())
  plt.plot([Y[:,d].min(), Y[:,d].max()],[Y[:,d].min(), Y[:,d].max()])
  plt.title('dim=%d' % d)

# boarder: huge jump!
# use sin and cos phi --> 10 dim!
# do prediction in the 10 dim space! - or modify the error
# turn two points back to phi for plotting


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
coords = [i[-1] for i in X]
dim1 = [i[0] for i in coords]
dim2 = [i[1] for i in coords]
dim3 = [i[2] for i in coords]
dim4 = [i[3] for i in coords]
dim5 = [i[4] for i in coords]

# loss = F.mse_loss
# errors = [loss(Yhat[i], Y[i]).item() for i in range(len(Y))]

# p = ax.scatter(xs, ys, zs, c = errors, cmap = 'viridis')
errors_plot = errors.copy()
errors_plot[errors_plot > .01] = .01
plt.scatter(dim1, dim2, dim3, c = errors_plot, cmap = 'plasma')

# https://matplotlib.org/stable/tutorials/colors/colormaps.html


plt.xlabel('time')
plt.ylabel('phases')
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
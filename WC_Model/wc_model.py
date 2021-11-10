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
import TimeSeriesTransformer as TST;

#%% Define WC Model

J = np.array([[15, -12],[15,-5]])
I = np.array([0.75, 0])

def g(x, b=1.1, th=1):
  return 1.0/(1 + np.exp(-4*b*(x-th)))

def model(x, t, J = J, I = I):
  dxdt = -x + J@g(x) + I
  return dxdt

#%% Plot Wilson-Cowan Model
# generate new trajectories starting at new points 
def initializeModel(ndim, length, samples):
  J = np.array([[15, -12],[15,-5]])
  I = np.array([0.75, 0])
  x0 = np.random.rand(ndim) # randomly initialized x0 - scale it to cover the limit cycles
  t = np.linspace(0, length, samples)
  x = si.odeint(model, x0, t)
  # generate x in a for loop (add another dimention for different initial condition!)
  plt.figure(1)
  plt.clf()
  plt.subplot(1,2,1)
  plt.plot(t, x)
  for i in range(1):
    plt.subplot(1,2,2+i)
    plt.imshow(J)
  return x
  # different x -> new y

ndim = 2
length = 100000
samples = 1000000
x = initializeModel(ndim, length, samples)
  

#%% Generate Data & Plot

def generate_data(x, tdim, odim, nsamples):
  # 1) pick a random trajectory 2) sample from the trajectory
  # [[x0=0.3] [x0=0.7] [] etc.]
  i = np.random.choice(np.arange(len(x)-odim-tdim),size=nsamples,replace=False)
  X=x[[np.arange(ii,ii+tdim) for ii in i]]
  Y=x[[np.arange(ii+tdim,ii+tdim+odim) for ii in i]]
  X = torch.tensor(X).float()
  Y = torch.tensor(Y).float()
  Y = torch.reshape(Y,(nsamples,-1))
  return X,Y

def plotData(X):
  plt.figure()
  plt.clf()
  for i in range(X.shape[0]):
    plt.plot(X[i,:,0], X[i,:,1])

tdim = 5
odim = 1
nsamples = 100000
X,Y = generate_data(x, tdim, odim, nsamples)
plotData(X)


# %% Timestep Evolution
def plotEvolution(na, X):
  fig = plt.figure() 
  fig.set_size_inches(10, 5)
  ax = fig.add_subplot(1, 2, 1)
  ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy())
  ax.scatter(X[0,:,0].detach().numpy(), X[0,:,1].detach().numpy(), c = 'red', s = 200) # Color with red!
  plt.show()

na = 500
plotEvolution(na, X)
#%%
reload(TST)

ndim = 2
tdim = 5
odim = 1
kdim = 2
vdim = 2
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

def train(x, model, optimizer, loss, epochs, iterations, nsamples):
  avg_err = 0
  for epoch in range(epochs):
    X,Y = generate_data(x, tdim, odim, nsamples=nsamples)
    l = train_batch(X, Y, model, optimizer, loss=loss, iterations=iterations)
    print('Epoch %d/%d: error: %r' % (epoch, epochs, l))
    if epoch + 10 >= epochs:
      avg_err += l
  avg_err = avg_err/10
  print('Average Training Error: ', avg_err)
  
learning_rate = 0.1

opt = optim.SGD(tst.parameters(), lr=learning_rate) # define optimizer
opt = optim.Adam(tst.parameters(), lr=1e-3)


# %%

train(x, tst, opt, loss, epochs=50, iterations = 1000, nsamples=100)

# %% Save Model
torch.save(tst.state_dict(), 'TST_WC_trained 2021-08-19' + '.pth')

#%% Load Network
tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
# tst.load_state_dict(torch.load('TST_Lorenz_trained 2021-08-03 10000-1000-10000.pth'))
tst.load_state_dict(torch.load('../../lorenz system/TST_WC_trained 2021-08-19 5-6-10000-1000-100-r-v.pth'))

tst.eval()
  

#%% Test Model and Print Error
def testModel(x, tst):
  X,Y = generate_data(x, 5, 1, 1000)
  Yhat = tst(X)
  loss = F.mse_loss
  l = loss(Yhat, Y)
  errors = np.sum(np.square(Y.detach().numpy() - Yhat.detach().numpy()), axis=-1)
  print('Test Set Error', l.item())
  if tdim == 1:
    f = open('tdim errors.txt', 'w+')
  else:
    f = open('tdim errors.txt', 'a+')
  f.write('tdim=' + str(tdim) + ':' + str(l.item()) + '\n')
  f.close() 
  plt.figure()
  plt.clf()
  for d in range(ndim):
    plt.subplot(1,ndim,1+d)
    plt.scatter(Y[:,d].detach().numpy(),Yhat[:,d].detach().numpy())
    plt.plot([Y[:,d].min(), Y[:,d].max()],[Y[:,d].min(), Y[:,d].max()])
    plt.title('dim=%d' % d)
  plt.figure()
  plt.clf()
  plt.hist(errors, bins=32)
  plt.show()
  return errors

errors = testModel(x, tst)

# %% Make Error Dot Plot
def errorDotPlot(X, errors):
  fig = plt.figure() 
  fig.set_size_inches(10, 8)
  plt.clf()
  coords = [i[-1] for i in X]
  xs = [i[0] for i in coords]
  ys = [i[1] for i in coords]
  errors_plot = errors.copy()
  errors_plot[errors_plot > .01] = .01
  plt.scatter(xs, ys, c = errors_plot, cmap = 'plasma')
  plt.xlabel('x-axis')
  plt.ylabel('y-axis')
  cbaxes = fig.add_axes([.95, .2, .05, .6])  # horizontal shift, vertical shift, bar width, bar height
  plt.colorbar(cax = cbaxes)

  plt.show()

errorDotPlot(X, errors)

# %% key-query plot prep
na = 500
X,Y = generate_data(x, tdim, odim, nsamples)
a = tst.attention(X)
print(a.shape)

# ref = query
# leg = key

def keyQueryPlot(na, x, tdim, odim):
  X,Y = generate_data(x, tdim, odim, nsamples=na)
  a = tst.attention(X)
  fig = plt.figure(25) 
  fig.set_size_inches(25, 20)
  for tref in range(tdim):
    for t in range(tdim):
      ax = fig.add_subplot(tdim, tdim, tdim*tref+t+1,)
      ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = a[:na,tref,t].detach().numpy(), cmap = 'plasma')
      plt.title("ref: %d  lag=%d" % (tdim-tref, tdim-t))
  fig.savefig('../WC_Model/images/attention_plot-'+'tdim:'+tdim+'.png', dpi=1000)

keyQueryPlot(na, x, tdim, odim)

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
    # p = ax.scatter(xs, ys, c = errors_plot, cmap = 'plasma')
    # plt.colorbar(p, cax = cbaxes)

# 1) generate a new window and rescale it - open a new matplotlib window instead of in-line plotting - plot in separate window
# 2) try less plots (as long as get the structures)

# fig.savefig('../images/wc-attention_plot_10-10000.png', dpi=1000)


# one plot: x points in orignals sampe

# %% Transform Plot Prep
na = 500
X,Y = generate_data(x, tdim, odim, nsamples=na)

X_arr = X.detach().numpy()[:,-1].copy()
for d in range(2):
  X_arr[:,d] -= X_arr[:,d].min()
  X_arr[:,d] *= 1/X_arr[:,d].max()
X_arr = np.concatenate([X_arr, np.zeros((X_arr.shape[0], 1))], axis=1)

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
ax = fig.add_subplot(1, 2, 1)
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = X_arr)

Q = tst.Q(X) 
ax = fig.add_subplot(1, 2, 2)
plt.title('Transform to Q')
ax.scatter(Q[:na,-1,0].detach().numpy(), Q[:na,-1,1].detach().numpy(), c = X_arr)
plt.show()

# %% V plot

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1)
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = X_arr)

V = tst.get_v(X[:na]).reshape(na, tdim, -1)
ax = fig.add_subplot(1, 2, 2)
plt.title('Transform to V')
ax.scatter(V[:na,-1,0].detach().numpy(), V[:na,-1,1].detach().numpy(), c = X_arr)
plt.show()
# %% R plot

fig = plt.figure() 
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 2, 1)
plt.title('Reference (X)')
ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = X_arr)

R = tst.get_r(X[:na]).reshape(na, tdim, -1)
ax = fig.add_subplot(1, 2, 2)
plt.title('Transform to R')
ax.scatter(R[:na,-1,0].detach().numpy(), R[:na,-1,1].detach().numpy(), c = X_arr)
plt.show()


# %%
# Plotting the results as training
# Plot the errors while training
# Save the plots in a picture file - look at the pictures! , .save_fig
# put a date on the files


# %% Main Method
def main():
    ndim = 2
    odim = 1
    kdim = 2
    vdim = 2
    fdim = 20
    length = 10000
    samples = 100000
    nsamples = 1000
    na = 500
    loss = F.mse_loss
    # learning_rate = 0.1
    x = initializeModel(ndim, length, samples)
    for i in range(1, 4):
      tdim = i
      X,Y = generate_data(x, tdim, odim, nsamples)
      plotData(X)
      plotEvolution(na, X)
      tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
      opt = optim.Adam(tst.parameters(), lr=1e-3)
      train(x, tst, opt, loss, epochs=50, iterations = 1000, nsamples=100)
      path = '/pth/TST_WC_trained 2021-08-17'+'n'+ndim+'o'+odim+'k'+kdim+'v'+vdim+'f'+fdim+'.pth'
      torch.save(tst.state_dict(), path)
      tst.load_state_dict(torch.load(path))
      errors = testModel(x, tst)
      errorDotPlot(X, errors)
      keyQueryPlot(na, x, tdim, odim)


if __name__ == "__main__":
    main()

# %%

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
from opts import *
import TimeSeriesTransformer as TST

J = np.array([[15, -12],[15,-5]])
I = np.array([0.75, 0])

def g(x, b=1.1, th=1):
  return 1.0/(1 + np.exp(-4*b*(x-th)))

def model(x, t, J=J, I=I):
  dxdt = -x + J@g(x) + I
  return dxdt

def initializeModel(ndim, length, timestep):
  x0 = np.random.rand(ndim) # randomly initialized x0
  t = np.linspace(0, length, int(length/timestep))
  x = si.odeint(model, x0, t)
  plt.figure(1)
  plt.clf()
  plt.subplot(1,2,1)
  plt.plot(t, x)
  for i in range(1):
    plt.subplot(1,2,2+i)
    # plt.imshow(J)
  return x

def generate_data(x, tdim, odim, nsamples):
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

def plotEvolution(na, X, tdim):
  fig = plt.figure() 
  fig.set_size_inches(10, 5)
  ax = fig.add_subplot(1, 2, 1)
  ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy())
  ax.scatter(X[0,:,0].detach().numpy(), X[0,:,1].detach().numpy(), c = 'red', s = 200) # Color with red!
  # plt.show()
  fig.savefig('../WC_Model/images/evolution_plot {} length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(date, length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)


def train_batch(x, y, model, optimizer, loss, iterations):
  for iteration in range(iterations):
    #print("Iteration %d / %d" % (epoch, epochs));
    optimizer.zero_grad()
    o = model(x)
    l = loss(o, y)
    l.backward()
    optimizer.step()

  return l.item()

def train(x, model, optimizer, loss, tdim, odim, epochs, iterations, nsamples, model_file):
  avg_err = 0
  for epoch in range(epochs): # use WHILE loop! - if change in error less than certain threhold, break (see if the system is improve - check last 10 trend)
    X,Y = generate_data(x, tdim, odim, nsamples=nsamples)
    l = train_batch(X, Y, model, optimizer=optimizer, loss=loss, iterations=iterations)
    print('Epoch %d/%d: error: %r' % (epoch, epochs, l))
    if epoch + 10 >= epochs:
      avg_err += l
  avg_err = avg_err/10
  print('Average Training Error: ', avg_err)
  torch.save(model.state_dict(), model_file)

def testModel(x, tst, ndim, tdim, odim, err_dict):
  X,Y = generate_data(x, tdim, odim, nsamples=nsamples)
  Yhat = tst(X)
  loss = F.mse_loss
  l = loss(Yhat, Y)
  errors = np.sum(np.square(Y.detach().numpy() - Yhat.detach().numpy()), axis=-1)
  print('Test Set Error', l.item())
  # if tdim == 1:
  #   f = open('tdim errors.txt', 'w+')
  # else:
  #   f = open('tdim errors.txt', 'a+')
  # f.write('tdim=' + str(tdim) + ':' + str(l.item()) + '\n')
  # f.close() 
  err_dict[tdim] = l.item()
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
  # plt.show()
  return errors, err_dict

def errorDotPlot(X, errors, tdim):
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
  # plt.show()
  fig.savefig('../WC_Model/images/error_plot {} length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(date, length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)


def keyQueryPlot(tst, na, x, tdim, odim):
  X,Y = generate_data(x, tdim, odim, nsamples=na)
  a = tst.attention(X)
  fig = plt.figure(25) 
  plt.clf()
  fig.set_size_inches(25, 20)
  for tref in range(tdim):
    for t in range(tdim):
      ax = fig.add_subplot(tdim, tdim, tdim*tref+t+1,)
      ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), c = a[:na,tref,t].detach().numpy(), cmap = 'plasma')
      plt.title("ref: %d  lag=%d" % (tdim-tref, tdim-t))
  # plt.show()
  fig.savefig('../WC_Model/images/attention_plot {} length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(date, length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)


def main():
    # learning_rate = 0.1
    print("Initialize Model")
    x = initializeModel(ndim, length, timestep)
    err_dict = {}
    for i in range(4, 5):
      tdim = i
      model_file = '../WC_Model/pth/TST_Wilson-Cowan_trained {} length={} timestep={} n={} t={} o={} k={} v={} f={}.pth'.format(date, length, timestep, ndim, tdim, odim, kdim, vdim, fdim)
      X,Y = generate_data(x, tdim, odim, nsamples)
      print("Plot Data")
      plotData(X)
      print("Plot Evolution")
      plotEvolution(na, X, tdim)
      print("construct Model")
      tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
      opt = optim.Adam(tst.parameters(), lr=learning_rate)
      print("Train Model")
      train(x, tst, opt, loss = F.mse_loss, tdim=tdim, odim=odim, epochs=50, iterations = 1000, nsamples=100, model_file=model_file)
      # delay --> if change in error is low --> stop ()
      # path = '/pth/TST_WC_trained 2021-08-17 n={} o={} k={} v={} f={}.pth'.format(ndim, odim, kdim, vdim, fdim)
      # torch.save(tst.state_dict(), path)
      # tst.load_state_dict(torch.load(path))
      print("Test Model")
      errors, err_dict = testModel(x, tst, ndim, tdim, odim, err_dict)
      print("errorDotPlot")
      errorDotPlot(X, errors, tdim)
      print("keyQueryPlot")
      keyQueryPlot(tst, na, x, tdim, odim)
    with open("../Lorenz_System/err/tdim errors {} length={} timestep={} n={} t={} o={} k={} v={} f={}.txt".format(date, length, timestep, ndim, tdim, odim, kdim, vdim, fdim), 'w') as f: 
        for key, value in err_dict.items():
            f.write('tdim={}:{} \n'.format(key, value))


if __name__ == "__main__":
    main()
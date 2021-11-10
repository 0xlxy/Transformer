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

def lorenz(x, t, sigma=10, rho=28, beta=8/3):
    y = np.zeros(3)
    y[0] = sigma * (x[1] - x[0])
    y[1] = x[0] * (rho - x[2]) - x[1]
    y[2] = x[0] * x[1] - beta * x[2]
    return y

def initializeModel(length, timestep): # generate x only once - use timestep (length/num_of_smaple) & length - simulate a long time series with a relatively small timestep
  # ex. change delta t to 2*delta t
    x0 = np.random.rand(3)
    t = np.linspace(0,length, int(length/timestep))
    x = si.odeint(lorenz, x0, t)
    plt.figure(1) # create figure object
    plt.clf()
    plt.plot(t, x)

    fig = plt.figure(2)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:,0], x[:,1], x[:,2])
    return x

def generate_data(x, tdim, odim, nsamples):
  # print("generate data inside")
  i = np.random.choice(np.arange(len(x)-odim-tdim),size=nsamples,replace=False)
  X=x[[np.arange(ii,ii+tdim) for ii in i]]
  Y=x[[np.arange(ii+tdim,ii+tdim+odim) for ii in i]]
  X = torch.tensor(X).float()
  Y = torch.tensor(Y).float()
  Y = torch.reshape(Y,(nsamples,-1))
  # print("X shape", X.shape)
  # print("Y shape", Y.shape)
  return X,Y

def plotEvolution(na, X, tdim):
  fig = plt.figure() 
  fig.set_size_inches(10, 5)
  ax = fig.add_subplot(1, 2, 1, projection='3d')
  ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy())
  ax.scatter(X[0,:,0].detach().numpy(), X[0,:,1].detach().numpy(), X[0,:,2].detach().numpy(), c = 'red', s = 300) # Color with red!
  # plt.show()
  print("end plot")
  fig.savefig('../Lorenz_System/images/2021-09-10 evolution_plot length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)

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
  for epoch in range(epochs):
    X, Y = generate_data(x, tdim, odim, nsamples=nsamples)
    l = train_batch(X, Y, model, optimizer=optimizer, loss=loss, iterations=iterations)
    # Store training errors (save every 10 epoch) - every 10 iteration --> store to the file
    print('Epoch %d/%d: error: %r' % (epoch, epochs, l))
    if epoch + 10 >= epochs:
      avg_err += l
  avg_err = avg_err/10
  print('Average Training Error: ', avg_err)
  torch.save(model.state_dict(), model_file)

def testModel(x, tst, ndim, tdim, odim, err_dict):
  # print("generate data")
  X,Y = generate_data(x, tdim, odim, nsamples=nsamples)
  # print("X shape", X.shape)
  # print("Y shape", Y.shape)
  Yhat = tst(X)
  # print("Yhat shape", Yhat.shape)
  loss = F.mse_loss
  l = loss(Yhat, Y)
  errors = np.sum(np.square(Y.detach().numpy() - Yhat.detach().numpy()), axis=-1)
  print('Test Set Error', l.item())
  # first store to a dictionary, and then store the distionary to the file!
  # if tdim == 1:
  #   f = open('../Lorenz_System/tdim errors.txt', 'w+')
  # else:
  #   f = open('../Lorenz_System/tdim errors.txt', 'a+')
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

# def errorDotPlot(X, errors, parameters):
def errorDotPlot(X, errors, tdim):
  fig = plt.figure() 
  fig.set_size_inches(10, 8)
  plt.clf()
  ax = fig.add_subplot(projection='3d')
  coords = [i[-1] for i in X]
  xs = [i[0] for i in coords]
  ys = [i[1] for i in coords]
  zs = [i[2] for i in coords]
  errors_plot = errors.copy()
  errors_plot[errors_plot > .01] = .01
  p = ax.scatter(xs, ys, zs, c = errors_plot, cmap = 'plasma')
  ax.set_xlabel('x-axis')
  ax.set_ylabel('y-axis')
  ax.set_zlabel('z-axis')
  cbaxes = fig.add_axes([.9, .2, .05, .6])  # horizontal shift, vertical shift, bar width, bar height
  plt.colorbar(p, cax = cbaxes)
  # plt.show()
  fig.savefig('../Lorenz_System/images/2021-09-10 error_plot length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)
  # fig.savefig('../Lorenz_System/images/error_plot-tdim={}.png'.format(parameters["tdim"]), dpi=1000)
  # parameters = {"tdim" : tdim, }

def keyQueryPlot(tst, na, x, tdim, odim):
  X,Y = generate_data(x, tdim, odim, nsamples=na)
  a = tst.attention(X)
  fig = plt.figure(25) 
  plt.clf()
  fig.set_size_inches(25, 20)
  for tref in range(tdim):
    for t in range(tdim):
      ax = fig.add_subplot(tdim, tdim, tdim*tref+t+1, projection='3d')
      ax.scatter(X[:na,-1,0].detach().numpy(), X[:na,-1,1].detach().numpy(), X[:na,-1,2].detach().numpy(), c = a[:na,tref,t].detach().numpy(), cmap = 'plasma')
      plt.title("ref: %d  lag=%d" % (tdim-tref, tdim-t))
  # plt.show()
  fig.savefig('../Lorenz_System/images/2021-09-10 attention_plot length={} timestep={} n={} t={} o={} k={} v={} f={}.png'.format(length, timestep, ndim, tdim, odim, kdim, vdim, fdim), dpi=1000)

def main():
    global tdim
    parameters = {"tdim": tdim, "kdim": kdim, "vdim": vdim, "fdim": fdim, "odim": odim} 
    # learning_rate = 0.1
    print("Initialize Model")
    x = initializeModel(length, timestep)
    err_dict = {}
    # First generate sample data
    # X,Y = generate_data(x=x, tdim=tdim, odim=odim, nsamples=nsamples) # Generate data once with a good delta t - get a long time series --> store to a file, and access everytime
    for i in range(3, 6):
      print("tdim = ", i)
      tdim = i
      model_file = '../Lorenz_System/pth/2021-09-10 TST_Lorenz_trained length={} timestep={} n={} t={} o={} k={} v={} f={}.pth'.format(length, timestep, ndim, tdim, odim, kdim, vdim, fdim)
      # Choose traing data from sample data
      X, Y = generate_data(x=x, tdim=tdim, odim=odim, nsamples=nsamples) # Generate data once with a good delta t - get a long time series --> store to a file, and access everytime
      # torch.save(X, '../Lorenz_System/X tdim={}.pt'.format(tdim))
      # torch.save(X, '../Lorenz_System/Y tdim={}.pt'.format(tdim))
      print("Plot Evolution")
      plotEvolution(na, X, tdim)
      print("construct Model")
      tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
      opt = optim.Adam(tst.parameters(), lr=learning_rate)
      print("Train Model")
      # def train(x, model, optimizer, loss, tdim, odim, epochs, iterations, nsamples, model_file):
      train(x, tst, opt, loss = F.mse_loss, tdim=tdim, odim=odim, epochs=50, iterations = 1000, nsamples=nsamples, model_file=model_file)
      # path = '/pth/TST_Lorenz_trained 2021-08-17 n={} o={} k={} v={} f={}.pth'.format(ndim, odim, kdim, vdim, fdim)
      # torch.save(tst.state_dict(), path)
      # print("Load Model")
      # tst.load_state_dict(torch.load(model_file))
      print("Test Model")
      errors, err_dict = testModel(x, tst, ndim, tdim, odim, err_dict)
      print("errorDotPlot")
      errorDotPlot(X, errors, tdim)
      print("keyQueryPlot")
      keyQueryPlot(tst, na, x, tdim, odim)
    with open("../Lorenz_System/err/tdim errors length={} timestep={} n={} t={} o={} k={} v={} f={}.txt".format(length, timestep, ndim, tdim, odim, kdim, vdim, fdim), 'w') as f: 
        for key, value in err_dict.items():
            f.write('tdim={}:{} \n'.format(key, value))

if __name__ == "__main__":
    main()
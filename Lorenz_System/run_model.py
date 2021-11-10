# %%
from python_model import *

# %%
def main():
    # learning_rate = 0.1
    print("Initialize Model")
    x = initializeModel(length, samples)
    # First generate sample data
    # X,Y = generate_data(x=x, tdim=tdim, odim=odim, nsamples=nsamples) # Generate data once with a good delta t - get a long time series --> store to a file, and access everytime
    for i in range(1, 4):
      print("tdim = ", i)
      tdim = i
      # Choose traing data from sample data
      X,Y = generate_data(x=x, tdim=tdim, odim=odim, nsamples=nsamples) 
      torch.save(X, '/Lorenz_System/X.pt')
      torch.save(Y, '/Lorenz_System/X.pt')

      print("Plot Evolution")
      plotEvolution(na, X, tdim)
      print("construct Model")
      tst = TST.TimeSeriesTransformer(ndim=ndim,tdim=tdim,kdim=kdim,vdim=vdim,fdim=fdim, odim=odim)
      opt = optim.Adam(tst.parameters(), lr=learning_rate)
      print("Train Model")
      train(x, tst, opt, loss = F.mse_loss, tdim=tdim, odim=odim, epochs=50, iterations = 1000, nsamples=100)
      path = '../Lorenz_System/pth/TST_Lorenz_trained 2021-09-01 n={} o={} k={} v={} f={}.pth'.format(ndim, odim, kdim, vdim, fdim)
      torch.save(tst.state_dict(), path)
      print("Load Model")
      tst.load_state_dict(torch.load(path))
      print("Test Model")
      errors = testModel(x, tst, ndim, tdim)
      errorDotPlot(X, errors, tdim)
      keyQueryPlot(tst, na, x, tdim, odim)

if __name__ == "__main__":
    main()
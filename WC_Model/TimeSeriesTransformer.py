#%% Construc attention network for time series analysis

import torch
import torch.nn as nn

# time ignorat version -> single k,q,v weights for every time step ?

#single key value pair mapping indepenent of ref 
class TimeSeriesTransformer(nn.Module):
  
  def __init__(self, ndim, tdim, kdim, vdim, fdim, odim, seed = 0):
    super(TimeSeriesTransformer, self).__init__()
    torch.manual_seed(seed)
    print("ndim: ", ndim)
    print("tdim: ", tdim)
    print("kdim: ", kdim)
    print("vdim: ", vdim)
    print("fdim: ", fdim)
    print("odim: ", odim)
    
    # Arguments
    # ---------
    # ndim : int
    #   Dimension of time series (data dimension)
    # tdim : int
    #   Number of time steps in the past (number of time points to consider in the past to predict future)
    # kdim : 
    #   Dimension of the key, querry pair represenation. (dimension for key and query vectors)
    # vdim : int
    #   Dimension of the represenation space (value vector dimension)
    # fdim : int
    #   Dimension of the neuronal network intermediate layers (number of nodes in the hiden layer of the feedforward network)
    # odim : int
    #   output time steps into the future (output vector dimension)
    
    # attention mechanism
    self.Q = nn.Linear(ndim, kdim)
    self.K = nn.Linear(ndim, kdim)
    self.V = nn.Linear(ndim, vdim)
    self.SMAX = nn.Softmax(dim=-1)
    print(self.Q)
    print(self.K)
    print(self.V)
    print(self.SMAX)
    
    # feed forward network
    self.FF = nn.Sequential( 
            nn.Linear(tdim * vdim, fdim), 
            nn.Sigmoid(),
            nn.Linear(fdim, fdim), 
            nn.Sigmoid(),
            nn.Linear(fdim, odim * ndim))
    print(self.FF)
    
    self.tdim = tdim
    self.ndim = ndim
    self.odim = odim
    self.vdim = vdim
  
  
  def attention(self, X):
    #self attention
    q = self.Q(X) # tst.Q.weight
    k = self.K(X)
    a = self.SMAX(torch.matmul(k, torch.transpose(q, -2,-1)))
    #dimensions: (nsamples, tdim (ref point, -1=last before next time step), tdim (sum over for represenation) )
    return a
  
  def forward(self, X):
    #self attention
    a = self.attention(X)
    v = self.V(X)
    
    #vector represenation
    r = torch.einsum('mij,mjk->mik', (a, v))
    r = torch.reshape(r, (-1,self.tdim*self.vdim))
    
    #feed forward ouput
    f = self.FF(r)
    #f = torch.reshape(f, (-1, self.odim, self.ndim));
    
    return f
  
  def get_v(self, X):
    a = self.attention(X)
    v = self.V(X)
  
    return v

  def get_r(self, X):
    a = self.attention(X)
    v = self.V(X)
    
    r = torch.einsum('mij,mjk->mik', (a, v))
    r = torch.reshape(r, (-1,self.tdim*self.vdim))
  
    return r
# %%

# inspect the project matrix - q, k, v - study the matricies with different initial conditions
# matrix = linear transformations - see how do they transform
# view as linear transformation & inspect proberties
# visualize: 3D grid, color regularlly, and plot the transformed points
# eigen value --> rank!
# eigen vector --> important directions!
# look at attention

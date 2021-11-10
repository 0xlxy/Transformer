date = '2021-09-14'
ndim = 2
tdim = 5
odim = 1
kdim = 2
vdim = 2
fdim = 20
length = 10000
timestep = 0.1
samples = 100000
nsamples = 1000
na = 500
learning_rate = 1e-3
parameters = {'ndim': 2,
              'tdim': [1, 10, 1],
              'odim': [1, 5, 1],
              'kdim': [1, 5, 1],
              'vdim': [1, 5, 1],
              'fdim': [5, 30, 5]}
train_on = ['tdim', 'odim', 'kdim', 'vdim', 'fdim']
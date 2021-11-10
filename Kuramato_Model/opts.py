date = '2021-09-26'
ndim = 5
tdim = 5
odim = 1
kdim = 2
vdim = 2
fdim = 20
length = 200
timestep = 0.01
nsamples = 1000
na = 500
learning_rate = 1e-3
parameters = {'ndim': [1, 5, 1],
              'tdim': [1, 10, 1],
              'odim': [1, 5, 1],
              'kdim': [1, 5, 1],
              'vdim': [1, 5, 1],
              'fdim': [5, 30, 5]}
train_on = ['ndim', 'tdim', 'odim', 'kdim', 'vdim', 'fdim']
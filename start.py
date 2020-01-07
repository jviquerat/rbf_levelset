# Generic imports
import os, sys

# Custom imports
from sampling    import *
from rbf_network import *

###############################################
### Set parameters
###############################################

# Basic parameters
n_basis     = 5         # nb of rbf functions to use
basis       = 'gaussian' # 'gaussian' or 'inv_mult'
normalize   = False       # normalized rbf if true
x_max       = 1.0        # bounds for function to fit
dim         = 2          # input dimension (1 or 2)
n_grid      = 100        # nb of grid evals per dimension for plotting

###############################################
### Fit function
###############################################

# Sample function
x = sample_random(0.5*x_max, dim, n_basis)
x = np.reshape(x, (n_basis, dim))
#y = sample_random(x_max, 1, n_basis)
#y = np.reshape(y, (n_basis))
y = np.ones(n_basis)

# Initialize network
network = rbf_network()
network.set(n_basis   = n_basis,
            basis     = basis,
            normalize = normalize,
            dim       = dim)

# Fill dataset
for i in range(n_basis):
    network.add_data(x[i,:], y[i])

# Train network
network.train()

# Generate grid
grid  = sample_grid(x_max, dim, n_grid)
y_net = np.reshape(network.predict(grid),(n_grid**dim,)) - 0.75

# Output rbf data
network.drop_rbf()
network.read_rbf(network.rbf_file)

# Output exact and approximate solutions on grid
filename  = 'grid_'+str(n_basis)+'.dat'
with open(filename, 'w') as f:
    for i in range(n_grid**dim):
        for k in range(dim):
            f.write('{} '.format(grid[i,k]))
        f.write('{} '.format(y_net[i]))
        f.write('\n')

# Generic imports
import os, sys

# Custom imports
from sampling    import *
from rbf_network import *

###############################################
### Set parameters
###############################################

# Basic parameters
n_basis     = 4         # nb of rbf functions to use
basis       = 'inv_mult' # 'gaussian' or 'inv_mult'
normalized  = False       # normalized rbf if true
sampling    = 'lhs'     # 'rand', 'grid' or 'lhs'
x_max       = 1.0        # bounds for function to fit
dim         = 2          # input dimension (1 or 2)
n_grid      = 50        # nb of grid evals per dimension for plotting

# Check command-line input for n_basis
if (len(sys.argv) == 2):
    n_basis = int(sys.argv[1])

###############################################
### Fit function
###############################################

# Sample function
x = sample_random(0.1*x_max, dim, n_basis)
x = np.reshape(x, (n_basis, dim))
y = sample_random(x_max, 1, n_basis)
y = np.reshape(y, (n_basis))

print(y)

# Initialize network
network = rbf_network(dim,
                      n_basis,
                      basis,
                      normalized)

# Fill dataset
for i in range(n_basis):
    network.add_data(x[i,:], y[i])

# Train network
network.train()

# Generate grid
grid  = sample_grid(x_max, dim, n_grid)

y_net = np.reshape(network.predict(grid),(n_grid**dim,))

# Output rbf data
filename  = 'dataset_'+str(n_basis)+'.dat'
with open(filename, 'w') as f:
    for i in range(n_basis):
        for k in range(dim):
            f.write('{} '.format(float(network.centers_x[i,k])))
        f.write('{} '.format(float(network.weights[i])))
        f.write('\n')

# Output exact and approximate solutions on grid
filename  = 'grid_'+str(n_basis)+'.dat'
with open(filename, 'w') as f:
    for i in range(n_grid**dim):
        for k in range(dim):
            f.write('{} '.format(grid[i,k]))
        f.write('{} '.format(y_net[i]))
        f.write('\n')

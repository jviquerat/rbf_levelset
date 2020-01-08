# Generic imports
import os, sys
import random

# Custom imports
from rbf_network import *

###############################################
### Set parameters
###############################################

# Basic parameters
n_basis     = 5          # nb of rbf functions to use
basis       = 'gaussian' # 'gaussian' or 'inv_mult'
normalize   = False      # normalized rbf if true
x_max       = 1.0        # bounds for function to fit
dim         = 2          # input dimension (1 or 2)
n_grid      = 100        # nb of grid evals per dimension for plotting
offset      =-0.75       # offset for level-set evaluation

###############################################
### Main execution
###############################################

# Regular execution, generate x/y, train and interpolate
if (len(sys.argv) == 1):

    # Generate samples
    x = np.zeros((n_basis, dim))
    y = np.ones(n_basis)

    for i in range(n_basis):
        for j in range(dim):
            x[i,j] = random.uniform(-0.25*x_max, 0.25*x_max)

    # Set network
    network = rbf_network()
    network.set_rbf(n_basis   = n_basis,
                    basis     = basis,
                    normalize = normalize,
                    dim       = dim,
                    x_max     = x_max,
                    offset    = offset,
                    n_grid    = n_grid)

    # Add data and train
    for i in range(n_basis):
        network.add_data(x[i,:], y[i])

    network.train()

    # Output rbf and dataset
    network.drop_rbf()
    network.drop_dataset()

    # Evaluate on grid
    network.eval_on_grid()

# Execute by reading file
if (len(sys.argv) == 2):

    # Get file name
    filename = sys.argv[1]

    # Set network from file
    network = rbf_network()
    network.read_rbf(filename)

    # Evaluate on grid
    network.eval_on_grid()

# Generic imports
import os
import numpy as np

###############################################
### Basic routines defining a grid
###############################################

# Mapping function from [0,1] to [-x_max,x_max]
def mapping(x, x_max):
    # Simplest mapping
    val = -x_max + 2.0*x_max*x
    return val

# Random sampling
def sample_random(x_max, dim, n_basis):
    x = np.random.rand(n_basis, dim)
    for i in range(n_basis):
        x[i,:] = mapping(x[i,:], x_max)

    return x

# Regular grid sampling
def sample_grid(x_max, dim, n_grid):

    # Generate a regular grid
    grid = np.zeros([n_grid**dim, dim])

    # Select dimension
    if (dim == 1):
        for i in range(n_grid):
            x = mapping(float(i)/float(n_grid-1), x_max)
            grid[i,0] = x

    if (dim == 2):
        for i in range(n_grid):
            x = mapping(float(i)/float(n_grid-1), x_max)
            for j in range(n_grid):
                y = mapping(float(j)/float(n_grid-1), x_max)
                grid[i*n_grid+j,0] = x
                grid[i*n_grid+j,1] = y

    return grid

# Uniform latin hypercube sampling
def sample_lhs(x_max, dim, n_samples):

    # Set random seed
    np.random.seed(1)

    # Generate 1D grid
    grid = sample_grid(x_max, 1, n_samples+1)
    x    = np.zeros([n_samples, dim])

    # Generate 1D LHS
    for i in range(n_samples):
        x[i,0] = np.random.uniform(low  = grid[i],
                                   high = grid[i+1])

    # If dim=2, generate a second 1D LHS
    if (dim == 2):
        grid = sample_grid(x_max, 1, n_samples+1)
        y    = np.zeros([n_samples])

        for i in range(n_samples):
            y[i] = np.random.uniform(low  = grid[i],
                                     high = grid[i+1])

        # Shuffle the second LHS
        np.random.shuffle(y)

        # Generate 2D LHS
        x[:,1] = y[:]

    return x

# Generic imports
import os
import numpy as np

###############################################
### Define rbf network class
###############################################

###############################################
### Class rbf_network
### A basic RBF network
class rbf_network:

    ### Create object
    def __init__(self):

        # Reset
        self.reset()

    ### Reset network
    def reset(self):

        # Reset network
        self.dim          = None
        self.basis        = None
        self.normalize    = None
        self.n_basis      = None
        self.centers_x    = None
        self.centers_y    = None
        self.weights      = None
        self.betas        = None
        self.x_max        = None
        self.offset       = None
        self.n_grid       = None
        self.trained      = None
        self.rbf_file     = None
        self.dataset_file = None
        self.dataset_inp  = np.array([])
        self.dataset_out  = np.array([])
        self.dataset_size = 0

    ### Set network from input
    def set_rbf(self, *args, **kwargs):

        # Handle arguments
        self.n_basis      = kwargs.get('n_basis',       5)
        self.basis        = kwargs.get('basis',         'gaussian')
        self.normalize    = kwargs.get('normalize',     False)
        self.dim          = kwargs.get('dim',           2)
        self.x_max        = kwargs.get('x_max',         1.0)
        self.offset       = kwargs.get('offset',       -0.5)
        self.n_grid       = kwargs.get('n_grid',        100)

        self.centers_x    = kwargs.get('centers_x',     None)
        self.centers_y    = kwargs.get('centers_y',     None)
        self.weights      = kwargs.get('weights',       None)
        self.betas        = kwargs.get('betas',         None)
        self.rbf_file     = kwargs.get('rbf_file',      None)

    ### Set dataset from input
    def set_dataset(self, *args, **kwargs):

        # Handle arguments
        self.dim          = kwargs.get('dim',           2)

        self.dataset_file = kwargs.get('dataset_file',  None)
        self.dataset_inp  = kwargs.get('dataset_inp',   np.array([]))
        self.dataset_out  = kwargs.get('dataset_out',   np.array([]))
        self.dataset_size = kwargs.get('dataset_size',  0)

    ### Rbf function
    def rbf(self, x, c, beta):

        # Compute radius
        r = np.linalg.norm(x-c)

        # Select rbf function
        if (self.basis == 'gaussian'):
            return np.exp(-beta*r**2)
        if (self.basis == 'inv_mult'):
            return np.sqrt(1.0/(1.0+beta*r**2))

    # Train network
    def train(self):

        # Compute centers and weights
        self.compute_centers()
        self.compute_betas()
        self.compute_weights()

        # Mark as trained
        self.trained = True

    ### Compute centers
    def compute_centers(self):

        # Select centers in the dataset
        # As of now, it is assumed that dataset size
        # is equal to nb of basis functions
        args = np.arange(self.n_basis)

        # Set centers
        self.centers_x = self.dataset_inp[args,:]
        self.centers_y = self.dataset_out[args]

    ### Compute betas
    def compute_betas(self):

        # Initialize array
        dist = np.zeros([self.n_basis,self.n_basis])

        # For each center, compute all distances
        for i in range(self.n_basis):
            for j in range(self.n_basis):
                a         = self.centers_x[i,:]
                b         = self.centers_x[j,:]
                dist[i,j] = np.linalg.norm(a-b)

        # Find closest neighbor for each center
        self.betas = np.zeros([self.n_basis])

        for i in range(self.n_basis):
            sigma = np.average(dist[i,:])
            self.betas[i] = 1.0/(sigma**2)

    ### Compute weights
    def compute_weights(self):

        # Generate matrix using centers vector
        matrix = self.compute_matrix(self.centers_x)

        # Solve resulting pseudo-linear system
        inv          = np.linalg.inv(matrix)
        self.weights = np.dot(inv, self.centers_y)

    # Predict once network is trained
    def predict(self, x):

        # Compute matrix and make prediction
        matrix = self.compute_matrix(x)
        y      = np.dot(matrix, self.weights)

        return y

    # Compute interpolation matrix
    def compute_matrix(self, x):

        # Interpolation matrix
        matrix = np.zeros([len(x), self.n_basis])

        for i in range(len(x)):
            for j in range (self.n_basis):
                matrix[i,j] = self.rbf(x[i],
                                       self.centers_x[j,:],
                                       self.betas[j])

        # Normalize if required
        if (self.normalize):
            for i in range(len(x)):
                norm         = np.sum(matrix[i,:])
                matrix[i,:] /= norm

        return matrix

    # Add data to the dataset
    # (x,y) should be a pair of (input,output)
    def add_data(self, x, y):

        # Stack into arrays
        if (self.dataset_size == 0):
            self.dataset_inp = x
            self.dataset_out = y
        else:
            self.dataset_inp = np.vstack((self.dataset_inp, x))
            self.dataset_out = np.vstack((self.dataset_out, y))
        self.dataset_size  += 1

    # Evaluate network on grid
    def eval_on_grid(self):

        # Generate grid
        grid = self.sample_grid()

        # Eval network on it
        y_net  = np.reshape(self.predict(grid),(self.n_grid**self.dim,))
        y_net += self.offset

        # Write grid file
        filename  = 'grid_'+str(self.n_basis)+'.dat'
        with open(filename, 'w') as f:
            for i in range(self.n_grid**self.dim):
                for k in range(self.dim):
                    f.write('{} '.format(grid[i,k]))
                f.write('{} '.format(y_net[i]))
                f.write('\n')

    # Drop dataset
    def drop_dataset(self):

        # Set filename
        filename          = 'dataset_'+str(self.n_basis)+'.dat'
        self.dataset_file = filename

        # Write dataset to file
        with open(filename, 'w') as f:
            f.write('{} '.format(self.dataset_size))
            f.write('{} '.format(self.dim))
            f.write('\n')
            for i in range(self.dataset_size):
                for j in range(self.dim):
                    f.write('{} '.format(self.dataset_inp[i,j]))
                f.write('{} '.format(float(self.dataset_out[i])))
                f.write('\n')

    # Read dataset
    def read_dataset(self,
                     filename):

        # Read header
        with open(filename) as f:
            line = f.read().split('\n')[0]
            line = line.rstrip()
            line = line.split(' ')

        # Read baseline info
        dataset_size = int(line[0])
        dim          = int(line[1])

        # Allocate arrays
        dataset_inp = np.zeros([dataset_size,dim])
        dataset_out = np.zeros([dataset_size])

        # Read dataset
        with open(filename) as f:
            content = f.read().split('\n')
            for i in range(1,dataset_size+1):
                j    = i-1
                line = content[i]
                line = line.rstrip()
                line = line.split(' ')

                for k in range(dim):
                    dataset_inp[j,k] = float(line[k])
                dataset_out[j] = float(line[dim + 0])

        # Set network
        self.set_dataset(dataset_size = dataset_size,
                         dim          = dim,
                         dataset_inp  = dataset_inp,
                         dataset_out  = dataset_out,
                         dataset_file = filename)

    # Drop rbf
    def drop_rbf(self):

        # Set filename
        filename      = 'rbf_'+str(self.n_basis)+'.dat'
        self.rbf_file = filename

        # Write file
        with open(filename, 'w') as f:
            f.write('{} '.format(self.n_basis))
            f.write('{} '.format(self.basis))
            f.write('{} '.format(self.normalize))
            f.write('{} '.format(self.dim))
            f.write('{} '.format(self.x_max))
            f.write('{} '.format(self.offset))
            f.write('{} '.format(self.n_grid))
            f.write('\n')
            for i in range(self.n_basis):
                for k in range(self.dim):
                    f.write('{} '.format(float(self.centers_x[i,k])))
                f.write('{} '.format(float(self.centers_y[i])))
                f.write('{} '.format(float(self.weights[i])))
                f.write('{} '.format(float(self.betas[i])))
                f.write('\n')

    # Read rbf
    def read_rbf(self,
                 filename):

        # Read header
        with open(filename) as f:
            line = f.read().split('\n')[0]
            line = line.rstrip()
            line = line.split(' ')

        # Read baseline info
        n_basis   = int(line[0])
        basis     = line[1]
        normalize = True if (line[2] == 'True') else False
        dim       = int(line[3])
        x_max     = float(line[4])
        offset    = float(line[5])
        n_grid    = int(line[6])

        # Allocate arrays
        centers_x = np.zeros([n_basis,dim])
        centers_y = np.zeros([n_basis])
        weights   = np.zeros([n_basis])
        betas     = np.zeros([n_basis])

        # Read centers, weights and betas
        with open(filename) as f:
            content = f.read().split('\n')
            for i in range(n_basis):
                line = content[i+1]
                line = line.rstrip()
                line = line.split(' ')

                for k in range(dim):
                    centers_x[i,k] = float(line[k])
                centers_y[i] = float(line[dim + 0])
                weights[i]   = float(line[dim + 1])
                betas[i]     = float(line[dim + 2])

        # Set network
        self.set_rbf(n_basis   = n_basis,
                     basis     = basis,
                     normalize = normalize,
                     dim       = dim,
                     x_max     = x_max,
                     offset    = offset,
                     n_grid    = n_grid,
                     centers_x = centers_x,
                     centers_y = centers_y,
                     weights   = weights,
                     betas     = betas,
                     rbf_file  = filename)

        self.trained = True

    # Regular grid sampling
    def sample_grid(self):

        # Generate a regular grid
        grid = np.zeros([self.n_grid**self.dim, self.dim])

        for i in range(self.n_grid):
            x = self.mapping(float(i)/float(self.n_grid-1), self.x_max)
            for j in range(self.n_grid):
                y = self.mapping(float(j)/float(self.n_grid-1), self.x_max)
                grid[i*self.n_grid+j,0] = x
                grid[i*self.n_grid+j,1] = y

        return grid

    # Simple mapping
    def mapping(self, x, x_max):

        val = -x_max + 2.0*x_max*x
        return val

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
    def __init__(self,
                 dim,
                 n_basis,
                 basis,
                 normalized):

        # Set sizes and rbf
        self.dim        = dim
        self.basis      = basis
        self.normalized = normalized

        # Set centers and weights
        self.reset(n_basis)

        # Dataset is an np.array containing the input/output
        # pairs for training
        self.dataset_inp  = np.array([])
        self.dataset_out  = np.array([])
        self.dataset_size = 0

    ### Reset network
    def reset(self,
              n_basis):

        # Reset weights and centers
        self.n_basis   = n_basis
        self.centers_x = None
        self.centers_y = None
        self.weights   = None
        self.betas     = None
        self.trained   = False

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

        # Reset weights and centers
        self.reset(self.n_basis)

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
                                       self.centers_x[j],
                                       self.betas[j])

        # Normalize if required
        if (self.normalized):
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

    # Drop dataset
    def drop_dataset(self):

        # Write inputs to file
        filename = 'dataset.input.dat'
        with open(filename, 'w') as f:
            for i in range(self.dataset_size):
                for j in range(self.dim):
                    f.write('{} '.format(self.dataset_inp[i,j]))
                f.write('\n')

        # Write outputs to file
        filename = 'dataset.output.dat'
        with open(filename, 'w') as f:
            for i in range(self.dataset_size):
                f.write('{} '.format(self.dataset_out[i]))
            f.write('\n')

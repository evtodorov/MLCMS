import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy import sparse


def spec_emd(data, n,  t = None, epsilon = None):
    '''
    Function to calculate the spectral embedding space of a given data set.
    The algorithm is inspired by the paper https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.9.5888
    by Mikhail Belkin and Partha Niyogi.


    The algorithm implements the three steps as described in the paper and returns n eigenfunctions which 
    can be utilised by the user.

    Inputs:

        :param data: (np.array with np.shape = (n_samples, n_features))
            Data to be embedded.

        :param n: (int)
            Number of eigenfunctions to return.

        :param t: (int)
            Parameter to use in exponent of the components in the adjacency graph W.

        :param epsilon: (int) (not implemented)
            Maximum Euclidean distance to add an edge in the adjacency graph W.


    Returns:

        :param eigenvectors: (np.array with np.shape = (n_samples, n))
            Reduced data, spctral embedding space
    '''    
    # Step 1, using (a).
    # Generate a adjacency graph W.
    W = distance_matrix(data, data)**2

    # Set constants epsilon and t if they are None
    if epsilon is None:
        epsilon = 0.01*np.max(W)

    if t is None:
        t = 1

    # Step 2, using (b)
    # Redo adjacency graph W with exponents.
    W = np.where(W < epsilon, np.exp(-W/t), 0)


    # Step 3 
    # Get laplacian L and make matrices sparse, return eigenvectors
    D = W@np.ones(W.shape[0])

    W = sparse.csr_matrix(W)
    D = sparse.diags(D, format = 'csr')
    L = D - W

    eigenvalues, eigenvectors = eigsh(L, n+1, which='SM')

    return eigenvectors[:,1:]

## Splitted functions for separated tesing ##

def adj_mat(data, epsilon = None, t=None):
    if t is None:
        t = 1

    
    # Step 1, using (a)
    W = distance_matrix(data, data)**2
    #plt.imshow(W)

    if epsilon is None:
        epsilon = 0.1*np.max(W)
    

    # Step 2, using (b)
    W = np.where(W < epsilon, np.exp(-W/t), 0)
    #np.fill_diagonal(W, 0)
    #W *= -t
    #W = np.exp(W)

    return W


def spec(W, n_components):
    n_nodes, m_nodes = W.shape
    D = W@np.ones(n_nodes)

    W = sparse.csr_matrix(W)
    D = sparse.diags(D, format = 'csr')
    laplacian = D - W


    eigenvalues, eigenvectors = eigsh(laplacian, n_components+1, which='SM')

    return eigenvectors[:,1:]


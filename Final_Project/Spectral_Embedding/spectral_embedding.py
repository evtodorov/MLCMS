import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
from scipy import sparse


def spec_emd(data, n_components, epsilon = None, t = None):
    if epsilon is None:
        epsilon = 1/data.shape[1]
    if t is None:
        t = 1/data.shape[1]

    
    # Step 1, using (a)
    W = distance_matrix(data, data)**2


    # Step 2, using (b)
    # D = np.where(D > epsilon, 0, np.exp(-D/t)) This unfortunately doesn't seen to work
    W *= -t
    W = np.exp(W)


    # Step 3 
    # Get laplacian and make matrices sparse
    D = W@np.ones(W.shape[0])

    W = sparse.csr_matrix(W)
    D = sparse.diags(D, format = 'csr')
    L = D - W


    eigenvalues, eigenvectors = eigsh(L, n_components+1, which='SM')

    return eigenvectors[:,1:]

## Splitted functions for separated tesing ##

def adj_mat(data, epsilon = None, t=None):
    if epsilon is None:
        epsilon = 1/data.shape[1]
    if t is None:
        t = 1/data.shape[1]

    
    # Step 1, using (a)
    D = distance_matrix(data, data)**2

    # Step 2, using (b)
    # D = np.where(D > epsilon, 0, np.exp(-D/t)) This unfortunately doesn't seen to work
    D *= -epsilon
    D = np.exp(D)

    return D


def spec(W, n_components, epsilon = None, t=None):
    n_nodes, m_nodes = W.shape
    D = W@np.ones(n_nodes)

    W = sparse.csr_matrix(W)
    D = sparse.diags(D, format = 'csr')
    laplacian = D - W


    eigenvalues, eigenvectors = eigsh(laplacian, n_components+1, which='SM')

    return eigenvectors[:,1:]


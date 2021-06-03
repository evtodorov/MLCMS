import math
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
import numpy as np

def diffusion_map(data, k):
    """
    Function that calculates and returns the eigenvalues and eigenvectors/functions of the
    Laplace applied on data. 
    
    The function follows the structure structure from the exercise sheet [Berry et al. 2013]
    and each step is marked with Step X.
    
    :param data: (np.array with np.shape = (dim, N))
        The data we wish to do reduce
    
    :param k: (int)
        Number of eigenvalues and eigenvectors/functions

    """
    N = np.shape(data)[0]

    # Step 1
    D = distance_matrix(data, data) 
    
    # Step 2
    epsilon = 0.05*np.max(D)

    # Step 3
    W = np.exp(-np.square(D)/epsilon)
    
    # Step 4
    P = W.sum(axis=0)*np.identity(N)
    
    # Step 5
    inv_P = np.linalg.inv(P) 
    K = np.dot(inv_P,W).dot(inv_P)

    # Step 6
    Q = K.sum(axis=0)*np.identity(N)
    
    # Step 7
    inv_Q_sq = np.zeros(Q.shape)
    np.fill_diagonal(inv_Q_sq, 1/(Q.diagonal()**0.5))
    T = np.dot(inv_Q_sq,K).dot(inv_Q_sq)
    
    # Step 8
    a_l, v_l = eigsh(T, k)

    # Step 9
    lamb = a_l**(1/(2*epsilon))
    
    # Step 10
    phi = np.dot(inv_Q_sq, v_l)

    return lamb, phi

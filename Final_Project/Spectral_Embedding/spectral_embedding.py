import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh


def spec_emd(data, n_components, epsilon = None, t=None):
    if epsilon is None:
        epsilon = 1/data.shape[1]
    if t is None:
        t = 1/data.shape[1]

    
    # Step 1, using (a)
    D = distance_matrix(data, data)


    # Step 2, using (b)
    D = np.where(D > epsilon, 0, np.exp(D/t))


    # Step 3 
    a_l, v_l = eigsh(D, n_components)

    return a_l, v_l



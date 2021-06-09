import numpy as np
def svd_of_matrix(X,rc,t_check=True):
    Xavg = np.mean(X, axis=rc)
    B = X - np.tile(Xavg, (X.shape[0], 1))

    if t_check == True:
        [U, S, VT] = np.linalg.svd((B.T) / np.sqrt(100))
    else:
        [U, S, VT] = np.linalg.svd((B) / np.sqrt(100))
    ret_values = [U, S, VT]
    return ret_values
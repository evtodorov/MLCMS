import numpy as np

def add_noise_dim(orig_data, no_noise_dim):

    no_data_points  = orig_data.shape[0]
    no_dim_orig_data = orig_data.shape[1]

    rng = np.random.default_rng(1)

    U_size = no_dim_orig_data + no_noise_dim
    Unormal = rng.normal(loc=0, scale=1, size=(U_size, U_size))

    U, S, VT = np.linalg.svd(Unormal)

    noise_shape = (no_data_points, no_noise_dim)
    noise_vals = np.random.normal(0, 0.5, noise_shape)

    stacked_data = np.hstack((orig_data, noise_vals))

    ret_data = stacked_data @ U

    return ret_data
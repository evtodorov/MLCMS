import numpy as np

def load_to_numpy(path):
    '''
    Loads and returns values x and f(x) stored column wise in the file under 'path'.
    x is assumed to be the first column and f(x) the second one.


    Input:
        :param path: (string)
            Path to the file where the requested values are stored
        
    Returns:
        :param x_es: (numpy array np.shape = (dim, 1)))
            Values of the first column in path

        :param f: (numpy array np.shape = (dim, 1)))
            Values of the second column in path

    ''' 
    
    try:
        data = np.loadtxt(path)
    except OSError as err:
        print(f'File in path {path} not found, error {err}')
        raise err
    
    print(f'Data loaded with shape {np.shape(data)}')

    x_es = data.T[0].reshape((len(data), -1))
    f = data.T[1].reshape((len(data), -1))

    print(f'Returning x_es of shape {np.shape(x_es)} and f(x) of shape {np.shape(f)}')

    return x_es, f


def linear_fit(x, f):
    '''
    Makes a linear fit to f(x) based on the values of x, i.e. makes
    the fit f(x) ~ f_hat(x) = k*x + m 

    Input:
        :param x: (numpy array np.shape = (dim, 1)))
            x values

        :param f: (numpy array np.shape = (dim, 1)))
            Function to be fitted
        
    Returns:
        :param k: (float)
            Linear coefficient of f_hat

        :param k: (float)
            Constant coefficient of f_hat

    '''

    A = np.vstack([x.T, np.ones(len(x))]).T
    k, m = np.linalg.lstsq(A, f, rcond=None)[0]
    print(f'Got coefficients for the fit f_hat(x) = k*x + m as \n k = {k[0].round(3)}, m = {m[0].round(3)}')
    return k, m


def non_linear_fit(x, f):
    pass
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
        :param f_hat: (numpy array np.shape = (dim, 1)))
            The linear fit to f(x) over the value of x

    '''

    A = np.vstack([x.T, np.ones(x.T.shape)]).T
    coeff = np.linalg.lstsq(A, f, rcond=None)[0]
    print(f'Got coefficients for the fit f_hat(x) = k*x + m, as \nk = {coeff[0][0].round(3)}, m = {coeff[1][0].round(3)}')
    f_hat = coeff[0]*x + coeff[1]

    return f_hat


def rbf_calc(x_l, x, eps):
    '''
    Calculates the radial basis function

    Input:
        :param x_l: (float)
            Centre of the basis function

        :param x: (numpy array np.shape = (dim, 1)))
            x values
        
        :param epsilon: (float)
            The bandwidth of the basis function
        
    Returns:
        :param rbf: (numpy array np.shape = (dim, 1)))
            Radial basis function of x, with parameters x_l and eps.

    '''

    r = (x_l - x)**2
    rbf = np.exp(-r/(eps**2))

    return rbf


def non_linear_fit(x, f, l, eps):
    '''
    Makes a nonlinear fit to f(x) based on the values of x,
    i.e. f(x) ~ f_hat(x) =  Σ cΦ

    Input:
        :param x: (numpy array np.shape = (dim, 1)))
            x values

        :param f: (numpy array np.shape = (dim, 1)))
            Function to be fitted

        :param l: (int)
            The number of radial basis functions to be combined

        :param epsilon: (float)
            The bandwidth of the basis function
        
        
    Returns:
        :param f_hat: (numpy array np.shape = (dim, 1)))
            The nonlinear fit to f(x) over the value of x

    '''
    
    N = x.shape[0]

    # Create evenly distributed x_ls
    x_ls = np.linspace(0, np.max(x), l).reshape(l,1) 

    # Fill a list with phis
    phi_sums = [rbf_calc(x_l, x, eps) for x_l in x_ls]
    phi_sums = np.array(phi_sums).reshape(l, N)
    
    # Get the coefficient
    A = np.vstack([phi_sums, np.ones(phi_sums.shape)]).T
    coeff = np.linalg.lstsq(A, f, rcond=None)[0]

    # Sum up all the radial basis functions
    func_lst = [rbf_calc(x_l, x, eps)*coeff[i] for i, x_l in enumerate(x_ls)]
    f_hat = np.sum(np.array(func_lst), axis=0)

    return f_hat
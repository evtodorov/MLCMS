# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

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

    x_train, x_test, f_train, f_test = train_test_split(x_es, f, test_size = 0.2, random_state = 1337)


    print(f'Length of training data: {len(x_train)} ({round(100*len(x_train)/len(x_es), 2)}%), validation data: {len(x_test)} ({round(100*len(x_test)/len(f), 2)}%).')

    return x_train, x_test, f_train, f_test


def linear_fit(x, f, rcond=0.005):
    '''
    Makes a linear fit to f(x) based on the values of x, i.e. makes
    the fit f(x) ~ f_hat(x) = k*x + m 

    Input:
        :param x: (numpy array np.shape = (dim, 1)))
            x values

        :param f: (numpy array np.shape = (dim, 1)))
            Function to be fitted
            
        :param rcond: (float) default 0.005
            Singular value cut-off threshold (rcond parameter for np.linalg.lstsq)
        
    Returns:
        :param f_hat: (numpy array np.shape = (dim, 1)))
            The linear fit to f(x) over the value of x

    '''

    A = np.vstack([x.T, np.ones(x.T.shape)]).T
    coeff = np.linalg.lstsq(A, f, rcond=rcond)[0]
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
    x_ls = np.linspace(np.min(x), np.max(x), l).reshape(l,1) 

    # Fill a list with phis
    phi_sums = [rbf_calc(x_l, x, eps) for x_l in x_ls]
    phi_sums = np.array(phi_sums).reshape(l, N)
    
    # Get the coefficient
    A = np.vstack([phi_sums, np.ones((1,N))]).T
    coeff = np.linalg.lstsq(A, f, rcond=0.005)[0]

    # Sum up all the radial basis functions
    func_lst = [rbf_calc(x_l, x, eps)*coeff[i] for i, x_l in enumerate(x_ls)]
    f_hat = np.sum(np.array(func_lst), axis=0) + coeff[-1] # Add constant term


    return f_hat


def MSE(tru, est):
    '''
    Compute the mean squared error
    
    :param tru: (np.array(shape = (n_samples, dim)))
        True values

    :param est: (np.array(shape = (n_samples, dim)))
        Estimated values
        
    :return: (float)
    '''
    tru = np.array([tru]).T if len(tru.shape) <= 1 else tru
    est = np.array([est]).T if len(est.shape) <= 1 else est
    return np.mean(np.linalg.norm(est-tru,axis=1)**2)


class RBF(BaseEstimator):
    def __init__(self, L, eps, periodic=False, rcond=None):
        '''
        Create a Radial Basis Function Predictor on random subset of the samples
        
        :param l: (int)
            The number of radial basis functions to be combined

        :param epsilon: (float)
            The bandwidth of the basis function
        
        :param periodic: (bool) default False
            Fit a periodic kernel, x should be normalized between 0 and π.
            The forecast will then also be periodic with period of π.

        :param rcond: (float) default None
            Singular value cut-off threshold (rcond parameter for np.linalg.lstsq)
        '''
        self.L = L
        self.eps = eps
        self.rcond = rcond
        self.periodic = periodic
        
    def fit(self, x, f):
        '''
        Train the predictor
        
        :param x: (np.array(shape = (n_samples, dim)))
            x values

        :param f: (np.array(shape = (n_samples, dim)))
            Function to be fitted
        '''
        # sort the data format
        assert(x.shape == f.shape)
        if len(x.shape) <= 1:
            x = np.array([x]).T 
            f = np.array([f]).T
        
        idx = sorted(np.random.choice(len(x), self.L, replace=False))
        self.xx = x[idx]
        phi = self._phi(x).T
        self.C, res, rank, svals = np.linalg.lstsq(phi, f, rcond=self.rcond)
        
    def predict(self, x):
        '''
        Predict 

        :param x: (np.array(shape = (n_samples, dim)))
            Predict x values

        :return f_hat: (np.array(shape = (n_samples, dim)))
            The nonlinear fit to f(x) over the value of x
        '''
        x = np.array([x]).T if len(x.shape) <= 1 else x
        phi = self._phi(x)
        f_hat = self.C.T@phi
        return f_hat.T
    
    def _phi(self, x):
        # Compute the RBF Gaussian kernel
        phi = np.zeros((self.L,len(x)))
        for i in range(self.L):
            r = (self.xx[i]-x)
            if self.periodic:
                phi[i] = np.exp(-np.sin(np.linalg.norm(r, axis=1))**2/self.eps**2)
            else:
                phi[i] = np.exp(-np.linalg.norm(r, axis=1)**2/self.eps**2)
        return phi    
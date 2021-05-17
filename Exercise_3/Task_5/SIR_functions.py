def mu(b, I, mu0, mu1):
    """
    Function that calculates and returns the recovery rate of infected persons.
    
    :param b: (float)
        The number of hospital beds per 10,000 persons

    :param I: (float)
        The number of infective persons
    
    :param mu0: (float)
        The minimum recovery rate

    :param mu1: (float)
        The maximum recovery rate

    """

    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Function that calculates and returns the basic reproduction number,
    i.e. how the infection spreads. The disease will be eliminated if R0 < 1.
    
    :param beta: (float)
        The average number of adequate contacts per unit time with infectious individuals.

    :param d: (float)
        The per capita natural death rate

    :param nu: (float)
        The per capita disease-induced death rate
    
    :param mu1: (float)
        The maximum recovery rate
    """

    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.

    :param I: (float)
        The number of infective persons

    :param mu0: (float)
        The minimum recovery rate

    :param mu1: (float)
        The maximum recovery rate

    :param beta: (float)
        The average number of adequate contacts per unit time with infectious individuals.

    :param A: (float)
        The recruitment rate of susceptibles (e.g. birth rate)

    :param d: (float)
        The per capita natural death rate

    :param nu: (float)
        The per capita disease-induced death rate

    :param b: (float)
        The number of hospital beds per 10,000 persons
    
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res  

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.

    :param t: (list[(int = 0), (int)])
        Start and end time for the model

    :param y: (list[(float), (float), (float)])
        Initial conditions for the model
        y[0] = S_0, number of susceptible persons
        y[1] = I_0, number of infective persons
        y[2] = S_0, number of susceptible persons
    
    :param mu0: (float)
        The minimum recovery rate

    :param mu1: (float)
        The maximum recovery rate

    :param beta: (float)
        The average number of adequate contacts per unit time with infectious individuals.

    :param A: (float)
        The recruitment rate of susceptibles (e.g. birth rate)

    :param d: (float)
        The per capita natural death rate

    :param nu: (float)
        The per capita disease-induced death rate

    :param b: (float)
        The number of hospital beds per 10,000 persons

    """

    S,I,R = y[:]
    tot_persons = S + I + R 
    m = mu(b, I, mu0, mu1)
    
    dSdt = A - S*d - (beta * S * I) / tot_persons
    dIdt = -I*(d + nu + m) + (beta * S * I) / tot_persons
    dRdt = m*I - d*R
    
    return [dSdt, dIdt, dRdt]
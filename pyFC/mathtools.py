"""
________________________________________________________________
Description

Some useful maths tools.

"""

import numpy as np


"""
________________________________________________________________
Classes

"""

class LogNormalPDF():
    """
    Theoretical lognormal distribution
    """
    def __init__(self, mu, sigma, gstat=False):
        """
        mu, sigma   mean, standard deviation
        gstat       bool. Whether mu and sigma are the statistics of the underlying Gaussian.

        Note for use of this class. To get statistical parameters, e.g. mu,
        mu_g, sigma, sigma_g, skew, kurt, median, mode, flatness, just access
        the desired member variable.
        """

        ## Gaussian and Lognormal means (m, mu) & standard deviations (s, sigma)
        if gstat == False:
            s = self.sigma_l2g(mu, sigma)
            m = self.mu_l2g(mu, sigma)
        else:
            s = sigma
            m = mu
            sigma = self.sigma_g2l(m, s)
            mean = self.mu_g2l(m, s)
        
        self.mu = mu
        self.mu_g = m
        self.sigma = sigma
        self.sigma_g = s

        ## higher moments
        self.skew = self.calc_skew()
        self.kurt = self.calc_kurt()
        self.median = self.calc_median()
        self.mode = self.calc_mode()
        self.flatness = self.calc_flatness()

    def mu_l2g(self, mu_l, sigma_l):
        return np.log(mu_l**2) - 0.5*np.log(mu_l**2 + sigma_l**2)

    def sigma_l2g(self, mu_l, sigma_l):
        return np.sqrt(np.log(mu_l**2 + sigma_l**2) - np.log(mu_l**2))

    def l2g(self, mu_l, sigma_l):
        return self.mu_l2g(mu_l, sigma_l), self.sigma_l2g(mu_l, sigma_l)

    def mu_g2l(self, mu_g, sigma_g):
        return np.exp(mu_g + sigma_g**2/2.)

    def sigma_g2l(self, mu_g, sigma_g):
        return mu_g2l(mu_g, sigma_g)*np.sqrt(np.exp(sigma_g**2) - 1)

    def g2l(self, mu_g, sigma_g):
        return self.mu_g2l(mu_g, sigma_g), self.sigma_g2l(mu_g, sigma_g)

    def _lgstats_sel(self, mu, sigma, gstat):
        """
        Returns gaussian mu, sigma. If gstat=False converts from lognormal mu,
        sigma. If mu = sigma = None, uses self.mu and self. sigma.

        mu, sigma  mean, standard deviation
        gstat       Whether mu and sigma are the statistics of the underlying Gaussian.
        """

        if gstat == False:
            if mu == None: mu = self.mu
            if sigma == None: sigma = self.sigma
            s = self.sigma_l2g(mu, sigma)
            m = self.mu_l2g(mu, sigma)
        else:
            if mu == None: mu = self.mu_g
            if sigma == None: sigma = self.sigma_g
            s = sigma
            m = mu
        
        return m, s

    def pdf(self, x, mu=None, sigma=None, gstat=False):

        m, s = self._lgstats_sel(mu, sigma, gstat)

        return np.exp(-(np.log(x) - m)**2/(2*s**2))/\
            (x*s*np.sqrt(2*np.pi)) 

    def cdf(self, x, mu=None, sigma=None, gstat=False):
        """
        Cumulative distribution function
        """

        import scipy.special as spsp

        m, s = self._lgstats_sel(mu, sigma, gstat)

        return 0.5 + 0.5*spsp.erfc((np.log(x) - m)/(np.sqrt(2)*s))

    def gpdf(self, logx, mu=None, sigma=None, gstat=False):
        """
        Corresponding Gaussian PDF. The first argument is called logx to indicate the use in
        this function as the distribution of the *natural* logarithm of x.

        Return 
        d log N / d log rho
        """

        m, s = self._lgstats_sel(mu, sigma, gstat)

        ## The PDF
        result = np.exp(-(logx - m)**2/(2*s**2))/\
                (s*np.sqrt(2*np.pi)) 

        ## Result is d N / d log rho
        return result

    def calc_mode(self, mu=None, sigma=None, gstat=False):
        m, s = self._lgstats_sel(mu, sigma, gstat)
        return np.exp(m - s*s)

    def calc_median(self, mu=None, sigma=None, gstat=False):
        m, s = self._lgstats_sel(mu, sigma, gstat)
        return np.exp(m)

    def calc_skew(self, mu=None, sigma=None, gstat=False):
        m, s = self._lgstats_sel(mu, sigma, gstat)
        return (np.exp(s*s) + 2)*np.sqrt(s*s - 1)

    def calc_kurt(self, mu=None, sigma=None, fisher=True, gstat=False):
        """
        This is the excess kurtosis
        fisher     return the Fisher version (gaussian -> 0)
                   else return the Pearson's version (gaussian -> 3)
        """
        m, s = self._lgstats_sel(mu, sigma, gstat)

        result = np.exp(4*s*s) + 2*np.exp(3*s*s) + 3*np.exp(2*s*s) - 6
        if fisher == False: result += 3

        assert(not(result < -2))
        return result

    def calc_flatness(self, mu=None, sigma=None, gstat=False):
        """
        See Appendix of Sutherland & Bicknell (2007)
        """
        m, s = self._lgstats_sel(mu, sigma, gstat)
        return self.calc_kurt(mu, sigma, fisher=False, gstat=gstat) + 3

"""
________________________________________________________________
Methods

"""

def moving_average(a, n=5, d=1) :
    ret = np.cumsum(a)
    return (ret[n-1:] - ret[:1-n])[::d]/n


def f_2f(x, f1, f2, xc, delta=1., damp_mode='switch'):
    """
    k            k array
    f1           array containing values of first function. Same size as k
    f2 
    damp_mode    'switch'   discontinuous switch
                 'roll'     exponential rollover
                 'tanh'     a tanh switcher
                 'ttanh'    a tanh switcher bounded by a tan function in its
                            argument
    """

    if damp_mode == 'switch':
        return np.where(x < xc, f1, f2)

    elif damp_mode == 'roll':
        return (f1 - f2)*np.exp(-x/xc) + f2

    elif damp_mode == 'tanh':
        return (0.5*(f1 - f2)*np.tanh(-(x - xc)/delta) + 0.5*(f1 + f2))

    elif damp_mode == 'ttanh':
        return np.where(x < xc-0.5*delta, f1, np.where(x > xc+0.5*delta, f2, 
               0.5*(f1 - f2)*np.tanh(np.tan(-np.pi*(x - xc)/
               (0.5*delta))) + 0.5*(f1 + f2)))
    else: 
        print('damp_mode "' + damp_mode + '" invalid')
        raise(ValueError)


def count_zero(a):
    """
    Useful function to count zeros in an array.
    """
    return np.count_nonzero(np.where(a == 0, 1, 0))

def count_nan(a):
    """
    Useful function to count zeros in an array.
    """
    return np.count_nonzero(np.where(np.isnan(a), 1, 0))

def count_inf(a):
    """
    Useful function to count zeros in an array.
    """
    return np.count_nonzero(np.where(np.isinf(a), 1, 0))


def zero_log10(s):
    """
    Takes logarithm of an array while retaining the zeros
    """
    sp = np.where(s>0.,s,1)
    return np.log10(sp)

def zero_div(a, b):
    """
    Function that "safely" divides by zero by retaining
    all values that were 0 as 0
    """
    bs = np.where(b>0, b, np.inf)
    return a/bs

def gaussian(x, m, s):
    return  np.exp(-(x - m)**2/(2*s**2))/(s*np.sqrt(2*np.pi))


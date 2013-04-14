# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as pl

## _______________________________________________
## Input parameters


## Dimensions of cube
ndim = 3
ni, nj, nk = shape = (64, 64, 64)

## Sampling limits (kmax = Nyquist limit)
## and power law index.
## (A faire: Make kmin and kmax ndim dimensional)
kmax = max(shape)/2 + 1
kmin = 4
beta = 5./3.

## Lognormal standard deviation and mean
sigma_l = np.sqrt(5.)
mean_l = 1.

## _______________________________________________
## Real bit

## Gaussian standard deviation and mean
sigma_g = np.sqrt(np.log(mean_l**2 + sigma_l**2) - np.log(mean_l**2))
mean_g = np.log(mean_l**2) - 0.5*np.log(mean_l**2 + sigma_l**2)

## Lognormal random field
lrf = np.random.lognormal(mean_g, sigma_g, shape)

## N-dimensional (all dimensional) FFT lognormal cube. 
F = fft.rfftn(lrf)

## k-space, 0-padded below kmin. Starting from 1.
k = np.arange(0, ni/2+1)

## The power spectrum is 4 pi k^2 F(k)* F(k) 
D = 4*np.pi*k**2*np.absolute(F)**2

## Index to omit 0 frequency of k. "From 1."
f1 = np.s_[1:]
def from1(a): return np.split(a, [1], axis=-1)[1]

## The power spectrum should be proportional to k^-beta for some
## turbulence model (e.g. beta=5/3 for Kolmogorov). 
## Normalize the power law, though.
pow = k[f1]**(-beta)
pow = np.where(k[f1] >= kmin, pow, 0)
pow /= np.trapz(pow)

## Apodize power spectrum
Df1 = from1(D)*pow

## Fourier transform back
F = np.sqrt(Df1/(4*np.pi*(k[f1])**2))
lrf_b = np.abs(fft.irfftn(F))

## Rho space and theoretical Lognormal pdf
rho_hist = np.arange(-4, 4.1, 0.1)
rho_pdf = np.arange(-4, 4., 0.1)

def LNpdf(rho, mu, sigma): 
    """
    Log-normal PDF.

    mu      mean of underlying Normal distribution
    sigma   std dev of underlying Normal distribution

    Return 
    d log N / d log rho

    Note
    The PDF (mathematically there is a rho in the 
    denominator, but in discrete code, the pdf is
    multiplied with rho to obtain a finite value of
    the probability in the range rho ±Δrho.
    """

    ## The PDF
    result = np.exp(-(np.log(rho) - mu)**2/(2*sigma**2))/\
            (sigma*np.sqrt(2*np.pi)) 

    ## Normalize it
    result /= np.trapz(result, np.log10(rho))

    ## Result is d log N / d log rho
    return result
  
"""
"""

## lrf_b and lrf don't have the same shape anymore

## Probability with which to change a cell in lrf_b, 
## whose denisty is over- (+) or underrepresented (-) 
## compared to the theoretically desired lognormal pdf 
#lnpdf = LNpdf(10**rho_pdf, mean_g, sigma_g)
#modpdf = np.histogram(np.log10(lrf_b), rho_hist, density=True)[0]

#corr_prob = (modpdf - lnpdf)/np.max(lnpdf, modpdf)

#lrf_b *= np.interp(lrf_b, rho_pdf, corr_ratio)

## create a reduction factor probability distriubiton for a given density

## Plot the original and modified lognmormal distributison
res, bins, ignored = pl.hist((np.log10(lrf), np.log10(lrf_b)), 
                             rho_hist, log=False, histtype='step', 
                             normed=True, align='mid')

pl.plot(rho_pdf, LNpdf(10**rho_pdf, mean_g, sigma_g))
pl.xlabel(r'$\log(\rho)$', size=18)
pl.ylabel(r'$d N/d \log(\rho)$', size=18)
pl.show(block=False)


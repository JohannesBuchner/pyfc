import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as pl
from mathtools import LogNormalPDF

"""
_________________________________________________________________________________
Description

This standalone (apart from the dependences above in the imported modules)
script creates a log-normal density cube from a by the iterative method of 
Lewis & Austin (2002)

A faire:
- Generalize to different ni, nj, nk
- Check that the routine works in 2D and 1D

"""


"""
_________________________________________________________________________________
Input parameters

"""

## Dimensions of cube
## Currently all need to be the same
ni, nj, nk = shape = (128, 128, 128)

## Sampling lower limit and power law index.
kmin = 4
beta = 5./3.

## Lognormal standard deviation and mean
sigma_l = np.sqrt(5.)
mean_l = 1.

## Iteration tolerance
iter_max = 10
iter_tol = 0.01

## The target spectra functions
def f_hk(k):
    """
    The high k funcitonal form
    """
    kp = np.where(k>0.,k,np.inf)
    return  kp**(-beta-2.)

def f_lk(k): 
    """
    The low k funcitonal form
    """
    ## constant to make f_hk and f_lk match at k=kmin
    c = kmin**(-beta-2.-4.)
    return c*k**4
    #return 0

damp_mode = 'tanh'
delta = 0.1
ldelta = np.log10((kmin+0.5*delta)/(kmin-0.5*delta))

## Do plots?
plot = False


"""
_________________________________________________________________________________
Helper functions, variables

"""

e2ten, ten2e = np.log10(np.e), np.log(10.)

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

def func_target_spec(k):
    """
    Just a readable shorthand for target spectrum function call f_2f
    """
    return f_2f(k, f_lk(k), f_hk(k), kmin, delta=delta, damp_mode=damp_mode)

def zero_log_spec(s):
    """
    Takes logarithm of a spectrum while retaining the zeros
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

def norm_target_spec(target_spec):
    """
    Normalize target spectrum so that apodizing 
    a random Gaussian cube with the root of target_spec
    does not change the mean and variance of the Gaussian

    Returns normalized spectrum.
    """
    shape, N = target_spec.shape, target_spec.size
    target_spec = np.ravel(target_spec)
    csqr = (N - 1.)/(pnorm(target_spec) - 1.)
    target_spec *= csqr; target_spec[0] = 1
    return target_spec.reshape(shape)

def iso_power_spec(kr, pspec_r, nk=ni, raveled=False, 
                   stdev=False, digitize=False):
    """
    kr        k array
    pspec_r   power spectrum array
    nk        size per dimension
    stdev     output stdvs as well

    kr and pspec must be an array of same shape

    We ravel the arrays because the digitize function requires 1d array

    If (already) raveled (at input), then assume that kr is 1d and that the 
    first element represents k0 + 1. If not, ravel and remove first elemnts 
    of both kr and pspec.
    """

    if not raveled: 

        ## Remember shape
        shape = kr.shape

        ## Ravel
        #kr = np.ravel(kr)[1:]
        #pspec_r = np.ravel(pspec_r)[1:]
        kr = np.ravel(kr)
        pspec_r = np.ravel(pspec_r)
        bins = np.append(np.unique(kr), kr.max()+1)

    psc, bins =  np.histogram(kr, bins) 
    psw, bins =  np.histogram(kr, bins, weights=pspec_r)
    means = psw/psc

    binc = 0.5*(bins[:-1] + bins[1:])

    if stdev or digitize: 
        psd = np.digitize(kr, bins[1:-1]) 

    if stdev:
        pss, bins =  np.histogram(kr, bins, weights=(pspec_r-means[psd])**2)
        stdvs = np.sqrt(pss/psc)


        if digitize: return means, binc, stdvs, psd.reshape(shape)
        else:        return means, binc, stdvs

    else:
        if digitize: return means, binc, psd.reshape(shape)
        else:        return means, binc

def plot_power_spectrum(k, ps, label, line, error=False, k0line=False,
                       reduce=True):
    """
    k           array of k values. If bin == False, then this
                is already the reduced k (that is, the k output
                from iso_power_spec)
    ps          power spectrum. If bin == False, then this is 
                alreday the reduced ps (that is the ps output
                from iso_power_spec)
    error       plot error bars too. Ths is obsolete
    k0line      plot a marker for value of k = 0
    reduce         bin data first: default is True
    Essentially plots output of iso_power_spec
    """

    if not error:
        if reduce: means, binc = iso_power_spec(k, ps)
        else: binc, means = k, ps 
        pl.plot(binc[1:], means[1:], line, label=label)

    else:
        if reduce:
            print('Cannot have bin and error.')
            raise(ValueError)
        means, binc, stdvs = iso_power_spec(k, ps, stdev=True)
        ylower = np.maximum(1, means - stdvs)
        yerr_lower = means - ylower
        pl.errorbar(binc[1:], means[1:], 
                    yerr=[yerr_lower[1:], stdvs[1:]])
    if k0line:
        pl.hlines(means[0], binc[0], binc[-1], line[0], 'dotted')

    pl.xscale('log')
    pl.yscale('log')

def plot_lrf(lrf):

    pl.imshow(np.log10(lrf[np.int(ni/2.),:,:]))

def plot_gpdf(grf, loge=True):
    """
    Plot the underlying gaussian distributions. Always plots 
    log10(rho) vs dN/dlog10(rho)

    grf          ndarray, old distribution 
    grf_corr     ndarray, corrected distribution 
    loge         bool, the input log density is base e 
    """

    ## Rho space arrays
    min, max, step = -4, 4.1, 0.1
    logrho_hist = np.arange(min, max, step)
    logrho_pdf = logrho_hist[0:-1] + 0.5*step

    ## Theoretical pdf
    gpdf = ln.gpdf(logrho_pdf*ten2e, mean_l, sigma_l)

    ## Gaussian data plot
    res, bins, ignored = pl.hist(np.ravel(grf*(e2ten if loge else 1)), 
                                 logrho_hist, log=False, 
                                 histtype='step', normed=True, align='mid',
                                 label='data' )

    pl.plot(logrho_pdf, gpdf*ten2e, label='target')
    pl.xlabel(r'$\log_{10}(\rho)$', size=18)
    pl.ylabel(r'$d N/d \log_{10}(\rho)$', size=18)
    pl.legend()

def plot_lpdf(lrf, lrf_corr):
    """
    Plot the lognormal distributions, rho vs dN/drho.

    lrf          ndarray, old distribution 
    lrf_corr     ndarray, corrected distribution 
    """

    ## Rho space arrays
    min, max, step = -4, 4.1, 0.1
    logrho_hist = np.arange(min, max, step)
    logrho_pdf = logrho_hist[0:-1] + 0.5*step
    rho_hist = 10**logrho_hist
    rho_pdf = 10**logrho_pdf

    ## Theoretical pdf
    lpdf = ln.pdf(rho_pdf, mean_l, sigma_l)

    ## Lognormal data plot
    res, bins, ignored = pl.hist((lrf, lrf_corr), 
                                 rho_hist, log=False, histtype='step', 
                                 normed=True, align='mid')

    pl.plot(rho_pdf, lnpdf)
    pl.ylim(0,1); pl.xlim(0,10)
    pl.xlabel(r'$\rho$', size=18)
    pl.ylabel(r'$d N/d \rho$', size=18)

    pl.show()

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

def convergence_estimator(rms, rms_old, i):
    dist, bins = np.histogram(lrf_corr, 60, density=True)
    binc = 0.5*(bins[:-1] + bins[1:])
    lnpdf = ln.pdf(binc, mean_g, sigma_g)
    rms_old = rms
    rms = np.sqrt(np.mean((dist - lnpdf)**2/lnpdf**2))
    print(str(i)+': convergence... '+str(rms_old)+' - '+str(rms)+' = '+str(rms_old - rms))
    return rms, rms_old

def power_spectrum(F, k=None):
    """
    The power spectrum is D = 4 pi k^2 F(k)* F(k) ~ k^-5/3
    or D ~ F(k)* F(k) ~ k^-11/3
    """
    #return 4*np.pi*np.abs(F)**2
    return np.real(np.conjugate(F)*F)

def pnorm(D):
    """
    D is power spectrum. This just sums all values assuming that 
    delta x (or delta k) in the power spectrum space is 1.
    """
    norm = np.sum(np.real(D))
    return norm 



"""
_________________________________________________________________________________
Main bit

"""

pl.ion()

## Theoretical lognormal object
ln = LogNormalPDF(mean_l, sigma_l)

## Convert to gaussian stats
mean_g, sigma_g = ln.l2g(mean_l, sigma_l)

## Gaussian random field
grf = np.random.normal(mean_g, sigma_g, shape)
lrf = np.exp(grf)

## Sampling space. See definition of FFT, k and frequency 
## at http://docs.scipy.org/doc/numpy/reference/routines.fft.html
sampling = fft.fftfreq(ni)
k1d = sampling*ni
ksqr = k1d*k1d
kmag = np.sqrt(np.add.outer(np.add.outer(ksqr, ksqr), ksqr))

## The target spectrum
target_spec = func_target_spec(kmag)
target_spec = norm_target_spec(target_spec)

## The apodization values
apod = np.sqrt(target_spec)

## N-dimensional (3-dimensional) FFT lognormal cube
Fg = fft.fftn(grf)
Fl = fft.fftn(lrf)  ## Only for plotting

## Apodize with power law
Fga = Fg*apod


## Plot stuff
if plot:

    ## Power spectra (Dg should be white noise)
    ## All of the below are only needed for plotting
    Dg = power_spectrum(Fg)
    Dga = power_spectrum(Fga)
    Dl = power_spectrum(Fl)

    fig, axs = pl.subplots(1, 3, figsize=(14,4))
    pl.sca(axs[0]); plot_lrf(np.real(lrf))
    pl.sca(axs[1]); plot_gpdf(np.real(grf))
    pl.sca(axs[2])
    plot_power_spectrum(kmag, Dg,'orig G', 'k:')
    plot_power_spectrum(kmag, Dga, 'apodized G', 'r-', k0line=True)
    plot_power_spectrum(kmag, Dl,'orig L', 'b-')
    pl.legend(loc=3)

## Loop begins here
convergence, iiter = 1, 1
while convergence > iter_tol and iiter <= iter_max:

    ## Fourier transform back (the imag part is negligible)
    grf_a = np.real(fft.ifftn(Fga))

    ## Create lognormal
    lrf_a = np.exp(grf_a)

    ## Power spectrum of lognormal is not desired power-law
    Fla = fft.fftn(lrf_a)
    Dla = power_spectrum(Fla)

    ## Isotropic power spectra
    Dla_iso, k_iso = iso_power_spec(kmag, Dla)

    ## Fits to the isotropic spectra
    lk_iso = np.log10(k_iso)

    ## zeroth order fit, to find best height for target spectrum 
    ## (kind of normalization)
    target_spec_iso = func_target_spec(k_iso)
    fit0c = np.polyfit(lk_iso, zero_log_spec(Dla_iso) -
                               zero_log_spec(target_spec_iso), 0)
    fit0 = fit0c[0] + zero_log_spec(target_spec_iso)

    ## Fit data with two polynomials k < kmin and k > kmin separately
    lkmin = np.log10(kmin)
    lkl, lkh = np.s_[lk_iso<lkmin], np.s_[lk_iso>=lkmin]
    fit2l = np.polyfit(lk_iso[lkl], zero_log_spec(Dla_iso[lkl]), 2)
    fit2h = np.polyfit(lk_iso[lkh], zero_log_spec(Dla_iso[lkh]), 2)
    p2l_iso = np.polyval(fit2l, lk_iso)
    p2h_iso = np.polyval(fit2h, lk_iso)
    fit2 = f_2f(lk_iso, p2l_iso, p2h_iso, lkmin, delta=ldelta, damp_mode=damp_mode)

    ## Plot data power-spectra
    if plot:

        ## Only needed for plot
        Dga_iso, k_iso = iso_power_spec(kmag, Dga)

        fig, axs = pl.subplots(1, 4, figsize=(18,4))
        pl.sca(axs[0]); plot_lrf(np.real(lrf_a))
        pl.sca(axs[1]); plot_gpdf(np.real(grf_a))
        pl.sca(axs[2]);
        plot_power_spectrum(k_iso, Dga_iso, 'apodized G', 'r:', k0line=True, reduce=False)
        plot_power_spectrum(k_iso, Dla_iso, 'apodized L', 'b:', k0line=True, reduce=False)
        pl.legend(loc=3)

        ## Plot the fits
        plot_power_spectrum(k_iso, 10**fit2, 'fit p2', 'k--', reduce=False)
        plot_power_spectrum(k_iso, 10**fit0, 'fit p0', 'g--', reduce=False)

        ## plot residuals
        pl.sca(axs[3])
        plot_power_spectrum(k_iso, zero_div(Dga_iso,target_spec_iso),
                            'apodized G', 'r:', reduce=False)
        plot_power_spectrum(k_iso, zero_div(Dla_iso,target_spec_iso),
                            'apodized L', 'b:', reduce=False)
        plot_power_spectrum(k_iso, zero_div(10**fit2,target_spec_iso),
                            'fit p2', 'k--', reduce=False)
        plot_power_spectrum(k_iso, zero_div(10**fit0,target_spec_iso),
                            'fit p0', 'g--', reduce=False)

    ## Corrections based on fits
    lkmag = zero_log_spec(kmag)
    p2l = np.polyval(fit2l, lkmag)
    p2h = np.polyval(fit2h, lkmag)
    p2 = f_2f(lkmag, p2l, p2h, lkmin, delta=ldelta, damp_mode=damp_mode)
    p0 = fit0c[0] + zero_log_spec(target_spec)
    corr = 0.4*(p0 - p2); corr[0,0,0] = 0

    ## Apply correction (in log space) to apodization spectrum
    corr_apod = apod*10**(corr)

    ## Re-Apodize with normalized power law
    ## From here, it's the same as before the loop.
    corr_apod2 = norm_target_spec(corr_apod**2)
    apod_old, apod = apod.copy(), np.sqrt(corr_apod2)
    Fga = Fg*apod
        
    ## Estimate convergence
    convergence = np.mean(zero_div(abs(power_spectrum(apod_old) - 
                                       power_spectrum(apod)),
                                  power_spectrum(apod))) 
    print('iteration ' + str(iiter))
    print('convergence = ' + str(convergence))
    print('')

    ## Get ready for next iteration
    iiter += 1


import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as pl
import mathtools as mt
import scipy.interpolate as si


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
## For 2D set nk = 1. For 1D, nj = nk = 1
ni, nj, nk = shape = (64, 64, 64)

## Sampling lower limit and power law index.
kmin = 8
beta = 5./3.

## Lognormal standard deviation and mean
sigma_l = np.sqrt(5.)
mean_l = 1.

## Iteration tolerance
iter_max = 10
iter_tol = 0.01
eta = 0.5

## The target spectra functions
def func_target_spec(k, kmin):
    """
    The high k>kmin funcitonal form. The value for k=0 is 
    determined by normalization.
    """
    kp = np.where(k>kmin, k, np.inf)
    return  kp**(-beta - 2.)

## Do plots? save file?
plot = True
save = False


"""
_________________________________________________________________________________
Helper functions, variables

"""

e2ten, ten2e = np.log10(np.e), np.log(10.)


def norm_target_spec(target_spec):
    """
    Normalize target spectrum so that apodizing 
    a random Gaussian cube with the root of target_spec
    does not change the mean and variance of the Gaussian

    Returns normalized spectrum.
    """
    shape, N = target_spec.shape, target_spec.size
    target_spec = np.ravel(target_spec)
    csqr = (N - 1.)/pnorm(target_spec[1:])
    target_spec *= csqr; target_spec[0] = 1
    return target_spec.reshape(shape)

def iso_power_spec(kr, pspec_r, raveled=False, 
                   stdev=False, digitize=False):
    """
    kr        k array
    pspec_r   power spectrum array
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

def plot_power_spec(k, ps, label, line, error=False, k0line=False,
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
            print('Cannot have "reduce" and "error."')
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
    """
    Plot lognormal density field midplane slice.
    Plot one-dimensional curve, if 1D field
    """

    if ndim > 1:
        pl.imshow(np.log10(lrf[:,:,np.int(nk/2.)]))

    else:
        pl.semilogy(lrf[:,0,0])

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

def power_spec(F, k=None):
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
ln = mt.LogNormalPDF(mean_l, sigma_l)

## Convert to gaussian stats
mean_g, sigma_g = ln.l2g(mean_l, sigma_l)

## Gaussian random field, and the corresponding lognormal random field
grf = np.random.normal(mean_g, sigma_g, shape)

## Sampling space. See definition of FFT, k and frequency 
## at http://docs.scipy.org/doc/numpy/reference/routines.fft.html
sampli, samplj, samplk = fft.fftfreq(ni), fft.fftfreq(nj), fft.fftfreq(nk)
k1di, k1dj, k1dk = sampli*ni, samplj*nj, samplk*nk
ksqri, ksqrj, ksqrk = k1di*k1di, k1dj*k1dj, k1dk*k1dk
ndim = 3 - mt.count_zero(np.array(shape)-1)
kmag = np.sqrt(np.add.outer(np.add.outer(ksqri, ksqrj), ksqrk))

## Isotropic k (kmag is also used as a dummy second argument)
dummy, k_iso = iso_power_spec(kmag, kmag)

## Some helpful arrays for log k
lk_iso = np.log10(k_iso); lkmin = np.log10(kmin)
lkmag = mt.zero_log(kmag)

## Some helpful indices that determine where fits and corrections are applied an. 
## Could also have created these in linear space. The important thing is that there 
## are two types. One for iso, and one for n-dimensional arrays.
sf_lk_iso = np.s_[lk_iso>=lkmin]
so_lk_iso = np.s_[np.logical_not(sf_lk_iso)]; so_lk_iso[0] = False
sf_lkmag = np.s_[lkmag>=lkmin]
so_lkmag = np.ravel(np.s_[np.logical_not(sf_lkmag)]); 
so_lkmag[0] = False; so_lkmag.reshape(shape)

## The target spectrum
target_spec = func_target_spec(kmag, kmin)
target_spec = norm_target_spec(target_spec)
target_spec_iso = func_target_spec(k_iso, kmin)

## The apodization values
apod = np.sqrt(target_spec)

## N-dimensional (3-dimensional) FFT lognormal cube
Fg = fft.fftn(grf)

## Apodize with power law
Fga = Fg*apod


## Plot stuff
if plot:

    ## Power spectra (Dg should be white noise)
    ## All of the below are only needed for plotting
    Dg = power_spec(Fg)
    Dga = power_spec(Fga)
    lrf = np.exp(grf)
    Fl = fft.fftn(lrf)
    Dl = power_spec(Fl)

    fig, axs = pl.subplots(1, 4, figsize=(18,4))
    pl.sca(axs[0]); plot_lrf(np.real(lrf))
    pl.sca(axs[1]); plot_gpdf(np.real(grf))
    pl.sca(axs[2])
    plot_power_spec(kmag, Dg,'orig G', 'k:')
    plot_power_spec(kmag, Dga, 'apodized G', 'r-', k0line=True)
    plot_power_spec(kmag, Dl,'orig L', 'b-')
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
    Dla = power_spec(Fla)

    ## Isotropic power spectra
    Dla_iso, k_iso = iso_power_spec(kmag, Dla)

    ## Fits to the isotropic spectra

    ## zeroth order fit, to find best height for target spectrum 
    ## (kind of normalization). The multiplication by 10**fit0
    ## retains zeroes in log space.
    weights = np.r_[np.diff(k_iso), np.diff(k_iso[-2:])]
    fit0 = np.average(mt.zero_log(Dla_iso[sf_lk_iso]) -
                      mt.zero_log(target_spec_iso[sf_lk_iso]),
                      weights=weights[sf_lk_iso])
    p0_iso = mt.zero_log(10**fit0*target_spec_iso)

    ## Fit power spec of lognormal with polynomials
    fit2 = np.polyfit(lk_iso[sf_lk_iso], mt.zero_log(Dla_iso[sf_lk_iso]), 2)
    p2_iso = np.polyval(fit2, lk_iso)

    ## Plot data power-spectra
    if plot:

        ## Power spectra. These are only needed for plotting
        Dga = power_spec(Fga)
        Dga_iso, k_iso = iso_power_spec(kmag, Dga)

        fig, axs = pl.subplots(1, 4, figsize=(18,4))
        pl.sca(axs[0]); plot_lrf(np.real(lrf_a))
        pl.sca(axs[1]); plot_gpdf(np.real(grf_a))
        pl.sca(axs[2]);
        plot_power_spec(k_iso, Dga_iso, 'apodized G', 'r:', k0line=True, reduce=False)
        plot_power_spec(k_iso, Dla_iso, 'apodized L', 'b:', k0line=True, reduce=False)
        pl.legend(loc=3)

        ## Plot the fits
        plot_power_spec(k_iso[sf_lk_iso], 10**p2_iso[sf_lk_iso], 'fit p2', 'k--', reduce=False)
        plot_power_spec(k_iso[sf_lk_iso], 10**p0_iso[sf_lk_iso], 'fit p0', 'g--', reduce=False)

        ## Plot residuals
        pl.sca(axs[3])
        plot_power_spec(k_iso, mt.zero_div(Dga_iso,target_spec_iso),
                        'apodized G', 'r:', reduce=False)
        plot_power_spec(k_iso, mt.zero_div(Dla_iso,target_spec_iso),
                        'apodized L', 'b:', reduce=False)
        plot_power_spec(k_iso[sf_lk_iso], mt.zero_div(10**p2_iso[sf_lk_iso],
                        target_spec_iso[sf_lk_iso]), 'fit p2', 'k--', reduce=False)
        plot_power_spec(k_iso[sf_lk_iso], mt.zero_div(10**p0_iso[sf_lk_iso],
                        target_spec_iso[sf_lk_iso]), 'fit p0', 'g--', reduce=False)


    ## Corrections based on fits. fit0 needs to be multiplied
    ## with the func_target_spec, otherwise 0s are not preserved.
    p2 = np.polyval(fit2, lkmag)
    p0 = mt.zero_log(10**fit0*func_target_spec(kmag, kmin))
    corr = np.where(sf_lkmag, eta*(p0 - p2), 0);

    ## Apply correction (in log space) to apodization spectrum
    corr_apod = apod*10**(corr)

    ## Re-Apodize with normalized power law
    ## From here, it's the same as before the loop.
    corr_apod2 = norm_target_spec(corr_apod**2)
    apod_old, apod = apod.copy(), np.sqrt(corr_apod2)
    Fga = Fg*apod


    ## Estimate convergence
    convergence = np.average(mt.zero_div(abs(power_spec(apod_old) - 
                                             power_spec(apod)),
                                         power_spec(apod)))
    print('iteration ' + str(iiter))
    print('convergence = ' + str(convergence))
    print('')

    ## Get ready for next iteration
    iiter += 1


## Save output file
if save:
    ## A last conversion

    ## Fourier transform back (the imag part is negligible)
    grf_a = np.real(fft.ifftn(Fga))

    ## Create lognormal
    lrf_a = np.exp(grf_a)

    ## Save file
    fname = ('k'+format(kmin,'0>2d')+'_'+
             format(ni,'0>4d')+
             ('x'+format(nj,'0>4d') if ndim>1 else '')+
             ('x'+format(nk,'0>4d') if ndim>2 else '')+
             '.dat')
    lrf_a.tofile(fname)


"""
_________________________________________________________________________________
Description

This file contains the main classes for creating and transforming fractal cubes.

Lognormal fractal cubes are constructed by the iterative method of 
Lewis & Austin (2002)

To do:
- Generalize to different ni, nj, nk. This requires 
  different kmin_i, kmin_j, kmin_k.
- Ensure that the routine works in 2D and 1D (Allow n{ijk}=1)
- Parallelize (with mpi4py?).

"""
import numpy as np
import numpy.fft as fft
import parsetools as pt
import os.path as osp
import mathtools as mt
from collections import OrderedDict
import glob

"""
_________________________________________________________________________________
Classes

"""

class FCSlicer():
    """
    Slices fractal cube
    """    
    def __init__(self):
        pass

    def slice(self, fc, ax=0, loc=0.5, scale='frac'):
        """ 
        fc       FractalCube object
        ax       axis perpendicular to slice plane
        loc      location of slice on that axis
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.

        Returns:   2D array of slice
        """

        if scale == 'frac': loc = np.round(loc*fc.cube.shape[ax])

        return np.rollaxis(fc.cube, ax)[loc,:,:]

    def tri_slice(self, fc, locs=(0.5,0.5,0.5), scale='frac'):
        """ 
        Description
        -----------
        A generator function for three perpendicular slices at point loc

        Arguments
        ---------
        fc       FractalCube object
        locs     location of tri_slice (point where planes intersect)
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.

        Yields
        ------
        Slice arrays for each axis and intercept.
        This will always yield three slices, regardless 
        of whether cube is 2D
        """

        if scale == 'frac': locs = np.round(locs*np.array(fc.cube.shape))

        for i, loc in enumerate(locs): 
            yield np.rollaxis(fc.cube, i)[loc,:,:]

class FCAffine():
    """
    Affine transformations on a fractal cube
    """
    def __init__(self):
        pass

    def translate(self, fc, ax=0, delta=0.5, scale='frac', out='copy'):
        """ 
        Translation with roll
        ax       axis along which to translate values
        delta    distance of translation
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.
        out      One of the modes accepted by fc._returner

        Returns:   translated data
        """


        if scale == 'frac': delta = int(np.round(delta*fc.cube.shape[ax]))

        result = np.roll(fc.cube, delta, axis=ax)
        return fc._returner(result, out)

    def permute(self, fc, choice=0, out='copy'):
        """ 
        Chooses one of six permutations of the cube's axis orders
        choice   choses one of six permuations:
            0: 012, 1: 210 = 0.T, 
            2: 120, 3: 021 = 2.T, 
            4: 201, 5: 102 = 4.T, 
            where T is the transpose
        out      One of the modes accepted by fc._returner

        Returns:   translated data
        """

        if choice%2 == 0: result = np.rollaxis(fc.cube , choice%3)
        else:             result = np.rollaxis(fc.cube , choice%3).T

        return fc._returner(result, out)

    def mirror(self, fc, ax=0, out='copy'):
        """
        Mirrors a cube about the midplane along axis ax

        out      One of the modes accepted by fc._returner
        """
        if ax == 0: result = fc.cube[::-1,:,:]
        if ax == 1: result = fc.cube[:,::-1,:]
        if ax == 2: result = fc.cube[:,:,::-1]

        return fc._returner(result, out)

class FCExtractor():
    """
    This class contains methods to perform cube extractions of a fractal cube.
    """
    def __init__(self, bg="small"):

        ## Capture the fractal cube object
        self.__dict__.update(locals()); del self.__dict__['self']

        ## ways to order extracted features
        self.orders = ['count', 'sum', 'mean', 'var', 'label']

    def extract_feature(self, fc, low=1.0,  order='count', rank=0, 
                        bgv=1.e-12, trim=False, out='copy'):
        """ 
        Uses np.ndimage.measurements.label to label features in fractal cube

        fc         Fractal cube object
        low        Lower threshold limit for extraction
        order      One of 'size', 'sum', 'var', 'extent', 'label'
                   Sets which order the features are given in and chosen
                   with *rank*:
                   'count'     size by number of pixels
                   'sum'       the sum of the values in the features
                   'mean'      size by mean of the values in the features
                   'var'  the varianace in the features
                   'label'     the default labelling
        rank       Which feature will be selected. The ordering is given by *mode*
        bgv        value for the pixels that are not the feature of the final
                   output. Default is a small number, 1.e-12.
        trim       False returns result with same dimensions as input fractal
                   cube. True returns the trimmed result from
                   ndimage.measurements.find_objects. (not yet programmed)
        out        One of the modes accepted by fc._returner

        Returns the extracted feature as a type determined by out

        To do:
        - Program trim
        """

        import scipy.ndimage.measurements as snm

        ## Threshold array to background values of zero 
        ##  and identify (label) all features
        lthr = self.lthreshold(fc, low, bgv=0, out='ndarray')
        labelled, nlabels = snm.label(lthr)

        ## Perform operation on all features with label comprehension
        ## Also, patch numpy counting function name

        assert(order in self.orders)

        if order == 'label':
            index = rank

        else:
            if order == 'count': order += '_nonzero'
            lbl_range = np.arange(1, nlabels+1)
            sizes = snm.labeled_comprehension(labelled, labelled, lbl_range, 
                                              getattr(np, order), int, 0)

            ## Get relevant index and create resulting cube
            index = np.argsort(sizes)[::-1][rank] + 1

        ## Create resulting ndarray
        result = np.where(np.equal(labelled, index), lthr, bgv)

        return fc._returner(result, out)

    def lthreshold(self, fc, low=1.0, bgv=0, out='copy'):
        """ 
        Apply lower value threshold to fractal cube 

        bgv     A value for the background (the points for which 
                the values were < low) 
        """

        result = np.where(np.less(fc.cube, low), bgv, fc.cube)
        return fc._returner(result, out)

class FCRayTracer():
    """ 
    This class performs raytrace operations
    """

    def __init__(self):
        pass

    def pp_raytrace(self, fc, absorb_coeff=1., emiss_coeff=1., ax=0):
        """
        Plane-parallel raytracer.

        Returns a 2D ndarray of a raytraced image given absorption and emission
        coefficients. This routine merely solves the radiative transfer
        integral from the back to the front of the box. We assume that:

            0. That we are integrating along axis=0.
            1. The box size is 1
            2. cell width are uniform and are a/ni
            3. there is no background source
            4. the emissivity is proportional to the cube variable
            5. the absorption coefficient is proportional to the 
               cube variable

        This raytracer is merely implemented for visualization purposes, which
        is the reason we have assumptions 4 and 5.
        """

        ## Emission and absorption coefficients
        c_abs = absorb_coeff*fc.cube
        c_emi = emiss_coeff*fc.cube

        ## Transmittance from one side of face to each cell
        ## Note: c_abs*delta gives optical depths
        ## Also make sure transmittance does not include current layer, so
        ## the array needs to be shifted by one layer toward the far boundary
        ## The near boundary layer should be filled with a plane of 1s.
        total_length = 1.
        if ax==0: delta = total_length/fc.ni
        if ax==1: delta = total_length/fc.nj
        if ax==2: delta = total_length/fc.nk

        transmittance = np.cumprod(np.exp(-c_abs*delta), axis=ax)
        if ax==0: transmittance = np.append(np.ones((1, fc.nj, fc.nk)), 
                                            transmittance[:-1,:,:], axis=ax)
        if ax==1: transmittance = np.append(np.ones((fc.ni, 1, fc.nk)), 
                                            transmittance[:,:-1,:], axis=ax)
        if ax==2: transmittance = np.append(np.ones((fc.ni, fc.nj, 1)), 
                                            transmittance[:,:,:-1], axis=ax)

        ## The resulting integrated result as a 2D array
        return np.sum(c_emi*(1. - np.exp(-c_abs*delta))*transmittance , axis=0)

class FCDataEditor():
    def __init__(self):
        pass

    def mult(self, fc, factor, out='copy'):
        result = fc.cube*factor
        return fc._returner(result, out)

    def pow(self, fc, power, out='copy'):
        result = (fc.cube)**power
        return fc._returner(result, out)

class FCStats():
    """
    Simple class that just collects statistical functions.
    useful for random fractal cube statistics.

    The benefit is that the returner function has the option 
    returning the modified fractal cube with the value of the 
    statistical parameter saved as an attribute, out='inplace', 
    a copy of the fractal cube, out='copy', or just the value,
    out='value'

    NOTE: UNTESTED!!
    """
    def __init__(self):

        import scipy.stats as sps
        self.__dict__.update(locals()); del self.__dict__['self']

    def mean(self, fc, out='value'):
        mean = np.mean(fc.cube)
        return self._stats_returner(fc, out, mean=mean)

    def std(self, fc, out='value'):
        std = np.std(fc.cube)
        return self._stats_returner(fc, out, std=std)

    def var(self, fc, out='value'):
        var = np.var(fc.cube)
        return self._stats_returner(fc, out, var=var)

    def median(self, fc, out='value'):
        median = np.median(fc.cube)
        return self._stats_returner(fc, out, median=median)

    def rms(self, fc, out='value'):
        rms = np.sqrt(np.mean(fc.cube**2))
        return self._stats_returner(fc, out, rms=rms)

    def skew(self, fc, out='value'):
        skew = sps.skew(fc.cube)
        return self._stats_returner(fc, out, skew=skew)

    def kurt(self, fc, out='value'):
        kurt = sps.kurtosis(fc.cube)
        return self._stats_returner(fc, out, kurt=kurt)

    def flatness(self, fc, out='value'):
        """
        See appendix in Sutherland & Bicknell 2007
        """
        return self._stats_returner(fc, out, kurt=kurt)

    def _stats_returner(fc, out, **kwargs):
 
        if out == 'inplace':
            for k, v in kwargs: setattr(fc, k, v)
            return fc
        elif out == 'copy':
            copy = fc.copy()
            for k, v in kwargs: setattr(copy, k, v)
            return copy
        elif out == 'value':
            return kwargs[0]
        elif out == 'values':
            return kwarg
        else:
            print('Invalid output mode')
            raise(ValueError)

class FractalCube():

    def __init__(self, ni=64, nj=64, nk=64, 
                 kmin=1, mean=1, sigma=5., beta=-5./3.):

        """
        arguments
        ---------
        n{i|j|k}    number of points in i, j, k directions
        kmin        kmin of cube
        mean        the mean of the single point distribution
        sigma       the variance of the single point distribution
        beta        the power of D(k), the power spectrum.

        some members
        ------------
        cube        The fractal cube data itself
        ndim        dimensionality of cube
        shape       [ni, nj, nk]

        References
        ----------
        Lewis & Austin (2002)
        """

        self.__dict__.update(locals()); del self.__dict__['self']

        ## Save some useful values
        self.shape = [ni, nj, nk]
        self.ndim = np.sum(np.greater(self.shape, 1))

        ## Dictionary of manipulators (just for bookkeeping)
        ## {name of object: [class, functions...]}
        self.manipulators = OrderedDict((
            ['fc_slicer',    [FCSlicer,    'slice', 'tri_slice']],
            ['fc_affine',    [FCAffine,    'translate', 'permute', 'mirror']],
            ['fc_extractor', [FCExtractor, 'extract_feature', 'lthreshold']],
            ['fc_raytracer', [FCRayTracer, 'pp_raytrace']]
        ))

        self._get_all_manipulators()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def read_cube(self, fname='data.dbl', prec='double'):
        """
        Read cube with file name fname

        fname  Data filename 

        prec   Floating point precision of output {'double'|'single'}
        """
        if prec == 'double':   dtype = np.dtype('<f8')
        elif prec == 'single': dtype = np.dtype('<f4')
        else: ValueError('Unknown prec '+prec)

        cube = np.fromfile(fname, dtype=dtype, count=-1)
        cube = cube.reshape(*self.shape)
        return self._returner(cube, 'ndarray')

    def write_cube(self, fname='data.dbl', app=True, pad=False, prec='double'):
        """
        Writes out a fractal cube data file in little endian, 
        double precision. Care is taken not to overwrite existing files.

        fname  Data filename 

        app    automatically append kmin, ni, nj, and nk values to filename.
               If app == True, append kmin, ni, nj, nk values to filename. 
               If suffixed files '<base>-[0-9][0-9]<ext>' exist, appended
               by a suffix one larger than the highest number found.

        pad    Pad the numbering with 0s

        prec   Floating point precision of output {'double'|'single'}
        """

        if prec == 'double':   dtype = np.dtype('<f8')
        elif prec == 'single': dtype = np.dtype('<f4')
        else: ValueError('Unknown prec '+prec)

        if app:
            ext = osp.splitext(fname)[1]
            base = osp.splitext(fname)[0]

            if pad:
                fname = (base+'_'+format(self.kmin,'0>2d')+'_'+
                         format(self.ni,'0>4d')+
                         ('x'+format(self.nj,'0>4d') if self.ndim>1 else '')+
                         ('x'+format(self.nk,'0>4d') if self.ndim>2 else '')+
                         ext)
            else:
                fname = (base+'_'+str(self.kmin)+'_'+format(self.ni)+
                         ('x'+str(self.nj) if self.ndim>1 else '')+
                         ('x'+str(self.nk) if self.ndim>2 else '')+
                         ext)
 
        fname = pt.unique_fname(fname, '-', '[0-9][0-9]')

        cube.tofile(fname, dtype=dtype)
        return 1

    def copy(self):
        """ Return a copy of this object """

        if self.cubetype == 'gaussian':
            return GaussianFractalCube(self.ni, self.nj, self.nk, 
                                       self.kmin, self.mean, self.sigma, self.beta)

        elif self.cubetype == 'lognormal':
            return LogNormalFractalCube(self.ni, self.nj, self.nk, 
                                       self.kmin, self.mean, self.sigma, self.beta)
        else:
            return FractalCube(self.ni, self.nj, self.nk, 
                               self.kmin, self.mean, self.sigma, self.beta)

    def _returner(self, result, out):
        """ 
        Given a result, return the correct thing given an out mode
        copy:    return a copy of the FractalCube object
        ndarray:     return array
        inplace: default, change in place
        """

        if out == 'inplace':
            self.cube = result
            return self
        elif out == 'ndarray':
            self.cube = result
            return result
        elif out == 'copy':
            copy = self.copy()
            copy.cube = result
            return copy
        else:
            print('Invalid output mode')
            raise(ValueError)

    def _get_all_manipulators(self):
        """
        Create methods in self from self.manipulators. The manipulator objects
        are also created and stored in self.
        """

        ## Create method attributes. 
        for o, m in self.manipulators.items():
            for f in m[1:]: setattr(self, f, self._m_curried(getattr(m[0](), f)))

    def _m_curried(self, m_func):
        """
        Currying function for creating method attributes whose first argument is
        the fractal cube, self, itself, rather an external cube instance.
        """
        def m_curry(*args, **kwargs): return m_func(self, *args, **kwargs)
        return m_curry

    def power_spec(self, cube=None):
        """
        Return power spectrum of this fractal cube
        """
        if cube == None: cube = self.cube
        return self._power_spec(fft.fftn(cube))

    def iso_power_spec(self, kmag=None, cube=None, stdev=False):
        """
        Return isotropic power spectrum of this fractal cube
        Returns tuple (D(|k|), |k|)
        """
        if kmag == None: kmag = self.kmag_sampling()
        if cube == None: cube = self.cube
        return self._iso_power_spec(kmag, self._power_spec(fft.fftn(cube)),
                                    stdev)

    def kmag_sampling(self, ni=None, nj=None, nk=None):
        """
        Sampling space. See definition of FFT, k and frequency 
        at http://docs.scipy.org/doc/numpy/reference/routines.fft.html
        """

        if ni == None: ni = self.ni
        if nj == None: nj = self.nj
        if nk == None: nk = self.nk

        sampli, samplj, samplk = fft.fftfreq(ni), fft.fftfreq(nj), fft.fftfreq(nk)
        k1di, k1dj, k1dk = sampli*ni, samplj*nj, samplk*nk
        ksqri, ksqrj, ksqrk = k1di*k1di, k1dj*k1dj, k1dk*k1dk
        kmag = np.sqrt(np.add.outer(np.add.outer(ksqri, ksqrj), ksqrk))

        return kmag

    def norm_spec(self, spectrum):
        """
        Normalize a spectrum so that apodizing 
        a random Gaussian cube with the root of spectrum
        does not change the mean and variance of the Gaussian

        Returns normalized spectrum.
        """
        shape, N = spectrum.shape, spectrum.size
        spectrum = np.ravel(spectrum)
        csqr = (N - 1.)/np.sum(np.real(spectrum[1:]))
        spectrum *= csqr; spectrum[0] = 1
        return spectrum.reshape(shape)

    def _iso_power_spec(self, kr, pspec_r, raveled=False, stdev=False, digitize=False):
        """
        kr        k array
        pspec_r   power spectrum array
        stdev     output stdvs as well
        digitize  output psd as well (even if stdev is false)

        kr and pspec must be an array of same shape

        We ravel the arrays because the digitize function requires 1d array

        If (already) raveled (at input), then assume that kr is 1d and that the 
        first element represents k0 + 1. If not, ravel and remove first elemnts 
        of both kr and pspec.
        """

        if not raveled: 
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

            if digitize: return means, binc, stdvs, psd.reshape(self.shape)
            else:        return means, binc, stdvs

        else:
            if digitize: return means, binc, psd.reshape(self.shape)
            else:        return means, binc

    def _power_spec(self, F, k=None):
        """
        Power spectrum. Without the factor of 4*pi.
        """
        return np.real(np.conjugate(F)*F)

class GaussianFractalCube(FractalCube):
    def __init__(self, ni=64, nj=64, nk=64, kmin=1, 
                 mean=0, sigma=200., beta=-5./3.):

        FractalCube.__init__(self, ni=ni, nj=nj, nk=nk, 
                             kmin=kmin, mean=mean, sigma=sigma, beta=beta)

        self.cubetype = 'gaussian'

    def func_target_spec(self, k, kmin=None, beta=None):
        """
        The target spectrum function
        """

        if kmin == None: kmin = self.kmin
        if beta == None: beta = self.beta

        kp = np.where(k>kmin, k, np.inf)
        return  kp**(beta - 2.)

    def gen_cube(self, history=False):
        """
        Generate a Gaussian fractal cube with the given statistics and 
        save it to file self.fname.

        If history == True, this routine returns the initial cube too.
        """

        ## Gaussian random field, and the corresponding lognormal random field
        grf = np.random.normal(self.mean, self.sigma, self.shape)

        ## Get the ndim kmag array (isotropic k)
        kmag = self.kmag_sampling()

        ## The target spectrum
        target_spec = self.func_target_spec(kmag)
        target_spec = self.norm_spec(target_spec)

        ## The apodization values
        apod = np.sqrt(target_spec)

        ## N-dimensional (3-dimensional) FFT lognormal cube
        Fg = fft.fftn(grf)

        ## Apodize with power law
        Fga = Fg*apod

        ## Fourier transform back (the imag part is negligible)
        cube = np.real(fft.ifftn(Fga))

        ## Return cubes. In "history mode", the cubes in the 
        ## beginning is also returned. The history mode is only
        ## used through the routine history_cubes
        if history: 
            return self._returner(grf, 'copy'), self._returner(cube, 'inplace')
        else: 
            return self._returner(cube, 'inplace')

    def lnfc(self):
        """
        Return the corresponding Lognormal field as a LogNormalFractalCube object
        """
        ln = mt.LogNormalPDF(self.mean, self.sigma, gstat=True)
        lnfc = LogNormalFractalCube(self.ni, self.nj, self.nk, ln.mu,
                                    ln.sigma, self.beta)
        lnfc.cube = np.exp(self.cube)
        return lnfc

class LogNormalFractalCube(FractalCube):
    def __init__(self, ni=64, nj=64, nk=64, kmin=1, 
                 mean=1, sigma=np.sqrt(5.), beta=-1.66666666666):

        FractalCube.__init__(self, ni=ni, nj=nj, nk=nk, 
                             kmin=kmin, mean=mean, sigma=sigma, beta=beta)

        self.cubetype = 'lognormal'

    def func_target_spec(self, k, kmin=None, beta=None):
        """
        The target spectrum function
        """

        if kmin == None: kmin = self.kmin
        if beta == None: beta = self.beta

        kp = np.where(k>kmin, k, np.inf)
        return  kp**(beta - 2.)

    def gen_cube(self, verbose=True):

        for fc in self.yield_cubes(history=False, verbose=verbose):
            return fc

    def yield_cubes(self, history=True, verbose=True):
        """
        Generate a lognormal fractal cube with the given statistics and 
        save it to file self.fname

        If history == True, this routine yields the cubes for each iteration.
        This option is actually not to be used. Use gen_cube instead.
        """

        ##_____________________________________
        ## Parameters (These can be changed)

        ## Iteration 
        self.iter_max = 10
        self.iter_tol = 0.01

        ## Correction factor
        self.eta = 0.5
    
        ##_____________________________________
        ## Main bit

        ## Theoretical lognormal object
        ln = mt.LogNormalPDF(self.mean, self.sigma)

        ## Convert to gaussian stats
        mean_g, sigma_g = ln.mu_g, ln.sigma_g

        ## Gaussian random field, and the corresponding lognormal random field
        grf = np.random.normal(mean_g, sigma_g, self.shape)

        ## Yield cube, if history is on
        if history: yield self._returner(np.exp(grf), 'copy')

        ## Get the ndim kmag array (isotropic k)
        kmag = self.kmag_sampling(self.ni, self.nj, self.nk)

        ## Isotropic k (kmag is also used as a dummy second argument)
        dummy, k_iso = self._iso_power_spec(kmag, kmag)

        ## Some helpful arrays for log k
        lkmag, lk_iso, lkmin = mt.zero_log10(kmag), np.log10(k_iso), np.log10(self.kmin)

        ## Some helpful indices that determine where fits and 
        ## corrections are applied an. These could have been defined in
        ## linear space, but this doesn't change the indexing. The important
        ## thing is that there is one for k_iso and one for kmag. sf refers to
        ## indices for fitting region, so, for the other region. 
        sf_lk_iso = np.s_[lk_iso>=lkmin]
        so_lk_iso = np.s_[np.logical_not(sf_lk_iso)]; so_lk_iso[0] = False
        sf_lkmag = np.s_[lkmag>=lkmin]
        so_lkmag = np.ravel(np.s_[np.logical_not(sf_lkmag)]); 
        so_lkmag[0] = False; so_lkmag.reshape(self.shape)

        ## The target spectrum
        target_spec = self.func_target_spec(kmag)
        target_spec = self.norm_spec(target_spec)
        target_spec_iso = self.func_target_spec(k_iso)

        ## The apodization values
        apod = np.sqrt(target_spec)

        ## N-dimensional (3-dimensional) FFT lognormal cube
        Fg = fft.fftn(grf)

        ## Apodize with power law
        Fga = Fg*apod

        ## Loop begins here
        convergence, iiter = 1, 1
        while convergence > self.iter_tol and iiter <= self.iter_max:

            ## Fourier transform back (the imag part is negligible)
            grf_a = np.real(fft.ifftn(Fga))

            ## Create lognormal
            lrf_a = np.exp(grf_a)

            ## Yield cube, if history
            if history: yield self._returner(lrf_a, 'copy')

            ## Power spectrum of lognormal is not desired power-law
            Fla = fft.fftn(lrf_a)
            Dla = self._power_spec(Fla)

            ## Isotropic power spectra
            Dla_iso, k_iso = self._iso_power_spec(kmag, Dla)

            ## Fits to the isotropic spectra

            ## zeroth order fit, to find best height for target spectrum 
            ## (kind of normalization). The multiplication by 10**fit0
            ## retains zeroes in log space.
            weights = np.r_[np.diff(k_iso), np.diff(k_iso[-2:])]
            fit0 = np.average(mt.zero_log10(Dla_iso[sf_lk_iso]) -
                              mt.zero_log10(target_spec_iso[sf_lk_iso]),
                              weights=weights[sf_lk_iso])
            p0_iso = mt.zero_log10(10**fit0*target_spec_iso)

            ## Fit power spec of lognormal with polynomials
            fit2 = np.polyfit(lk_iso[sf_lk_iso], mt.zero_log10(Dla_iso[sf_lk_iso]), 2)
            p2_iso = np.polyval(fit2, lk_iso)

            ## Corrections based on fits. fit0 needs to be multiplied
            ## with the func_target_spec, otherwise 0s are not preserved.
            p2 = np.polyval(fit2, lkmag)
            p0 = mt.zero_log10(10**fit0*self.func_target_spec(kmag))
            corr = np.where(sf_lkmag, self.eta*(p0 - p2), 0);

            ## Apply correction (in log space) to apodization spectrum
            corr_apod = apod*10**(corr)

            ## Re-Apodize with normalized power law
            ## From here, it's the same as before the loop.
            corr_apod2 = self.norm_spec(corr_apod**2)
            apod_old, apod = apod.copy(), np.sqrt(corr_apod2)
            Fga = Fg*apod


            ## Estimate convergence
            convergence = np.average(mt.zero_div(abs(self._power_spec(apod_old) - 
                                                     self._power_spec(apod)),
                                                 self._power_spec(apod)))
            ## Some messages
            if verbose:
                print('iteration ' + str(iiter))
                print('convergence = ' + str(convergence))
                print('')

            ## Get ready for next iteration
            iiter += 1

        ## A last conversion

        ## Fourier transform back (the imag part is negligible)
        grf_a = np.real(fft.ifftn(Fga))

        ## Create lognormal
        cube = np.exp(grf_a)

        ## Return cube object
        yield self._returner(cube, 'inplace')

    def gnfc(self):
        """
        Return the corresponding Gaussian field as a GaussianFractalCube object
        """
        ln = mt.LogNormalPDF(self.mean, self.sigma)
        gnfc = GaussianFractalCube(self.ni, self.nj, self.nk, ln.mu_g,
                                   ln.sigma_g, self.beta)
        gnfc.cube = np.log(self.cube)
        return gnfc


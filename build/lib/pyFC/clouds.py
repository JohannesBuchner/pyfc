"""
.. module:: clouds
   :synopsis: Sub-module containing all the main classes for creating and transforming fractal cubes.


.. moduleauthor:: Alexander Y. Wagner <alexaner.y.wagner@gmail.com>


To do:

  - Ensure that the routine works in 2D and 1D (Allow n{ijk}=1)
  - Parallelize (with mpi4py?).

"""
import numpy as np
import numpy.fft as fft
import parsetools as pt
import os.path as osp
import copy
import mathtools as mt
from collections import OrderedDict


# TODO Ensure that the routine works in 2D and 1D (Allow n{ijk}=1)
# TODO Parallelize (with mpi4py?).

class FCSlicer():
    """
    Slices fractal cube object.
    """

    def __init__(self):
        pass

    def slice(self, fc, ax=0, loc=0.5, scale='frac'):
        """
        A simple mid-plane slice.

        :arg obj fc:        :class:`pyFC.FractalCube` object
        :arg int ax:        Axis number perpendicular to slice plane
        :arg int loc:       Location of slice on that axis
        :arg str scale:     {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to nearest integer,
                            frac) for location. Default 'frac'.

        :return:            2D array of slice
        :rtype:             ndarray
        """

        if scale == 'frac': loc = np.round(loc * fc.cube.shape[ax])

        return np.rollaxis(fc.cube, ax)[loc, :, :]

    def tri_slice(fc, locs=(0.5, 0.5, 0.5), scale='frac'):
        """ 
        A generator function for three perpendicular slices at point loc

        :arg obj fc:        :class:`pyFC.FractalCube` object
        :arg tup locs:      3-Tuple indicating location of tri-slice (point where planes intersect)
        :arg str scale:     {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to nearest integer,
                            frac) for location. Default 'frac'.

        :return:            Yield slice arrays for each axis and intercept. This will always yield three
                            slices, regardless of whether cube is 2D
        :rtype:             ndarray
        """

        if scale == 'frac': locs = np.round(locs * np.array(fc.cube.shape))

        for i, loc in enumerate(locs):
            yield np.rollaxis(fc.cube, i)[loc, :, :]


class FCAffine():
    """
    Affine transformations on a fractal cube
    """

    def __init__(self):
        pass

    def translate(self, fc, ax=0, delta=0.5, scale='frac', out='copy'):
        """ 
        Translation of cube using np.roll.

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg int ax:        Axis number along which to translate.
        :arg flt delta:     Distance of translation.
        :arg flt scale:     {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                            nearest integer, frac) for location. Default 'frac'.
        :arg str out:       One of the modes accepted by fc._returner.

        :return:            Translated cube
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.
        
        """

        if scale == 'frac': delta = int(np.round(delta * fc.cube.shape[ax]))

        result = np.roll(fc.cube, delta, axis=ax)
        return fc._returner(result, out)

    def permute(self, fc, choice=0, out='copy'):
        """ 
        Chooses one of six permutations of the cube's axis orders.

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg int choice:    Choses one of six permuations::

                                0: 012, 1: 210 (= 0.T), 
                                2: 120, 3: 021 (= 2.T), 
                                4: 201, 5: 102 (= 4.T), 

                            where T is the transpose.

        :arg str out:       One of the modes accepted by fc._returner.

        :return:            Permuted cube.
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.
 
        """

        if choice % 2 == 0:
            result = np.rollaxis(fc.cube, choice % 3)
        else:
            result = np.rollaxis(fc.cube, choice % 3).T

        return fc._returner(result, out)

    def mirror(self, fc, ax=0, out='copy'):
        """
        Mirrors a cube about a face along axis ax.

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg int ax:        Axis number perpendicular to which the mirror plane lies.
        :arg str out:       One of the modes accepted by fc._returner.

        :return:            Mirrored cube
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.

        """
        if ax == 0: result = fc.cube[::-1, :, :]
        if ax == 1: result = fc.cube[:, ::-1, :]
        if ax == 2: result = fc.cube[:, :, ::-1]

        return fc._returner(result, out)


class FCExtractor():
    """
    This class contains methods to perform cube extractions of a fractal cube.
    """

    def __init__(self, bg="small"):

        # Capture the fractal cube object
        self.__dict__.update(locals());
        del self.__dict__['self']

        # ways to order extracted features
        self.orders = ['count', 'sum', 'mean', 'var', 'label']

    def extract_feature(self, fc, low=1.0, order='count', rank=0,
                        bgv=1.e-12, trim=False, out='copy'):
        """ 
        Uses np.ndimage.measurements.label to "label" features in fractal cube.

        :arg obj fc:         :class:`pyFC.FractalCube` object.
        :arg flt low:        Lower threshold limit for extraction
        :arg str order:      One of 'size', 'sum', 'var', 'extent', 'label'
                             Sets which order the features are given in and chosen
                             with *rank*:

                             -'count':     size by number of pixels
                             -'sum':       the sum of the values in the features
                             -'mean':      size by mean of the values in the features
                             -'var':       the varianace in the features
                             -'label':     the default labelling

        :arg int rank:       Which feature will be selected. The ordering is given by *mode*
        :arg flt bgv:        value for the pixels that are not the feature of the final
                             output. Default is a small number, 1.e-12.
        :arg bool trim:      False returns result with same dimensions as input fractal
                             cube. True returns the trimmed result from
                             ndimage.measurements.find_objects. (Not yet programmed.)
        :arg str out:        One of the modes accepted by fc._returner.

        :return:             The extracted feature
        :rtype:              Type depends on 'out'. By default it returns a copy
                             of the :class:`pyFC.FractalCube` object.

        """

        import scipy.ndimage.measurements as snm

        # Threshold array to background values of zero 
        #  and identify (label) all features
        lthr = self.lthreshold(fc, low, bgv=0, out='ndarray')
        labelled, nlabels = snm.label(lthr)

        # Perform operation on all features with label comprehension
        # Also, patch numpy counting function name

        assert (order in self.orders)

        if order == 'label':
            index = rank

        else:
            if order == 'count': order += '_nonzero'
            lbl_range = np.arange(1, nlabels + 1)
            sizes = snm.labeled_comprehension(labelled, labelled, lbl_range,
                                              getattr(np, order), int, 0)

            # Get relevant index and create resulting cube
            index = np.argsort(sizes)[::-1][rank] + 1

        # Create resulting ndarray
        result = np.where(np.equal(labelled, index), lthr, bgv)

        return fc._returner(result, out)

    def lthreshold(self, fc, low=1.0, bgv=0, out='copy'):
        """ 
        Apply lower value threshold to fractal cube 

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg flt low:       Lower threshold limit for extraction
        :arg flt bgv:       A value for the background (the points for which 
                            the values were < low) 
        :arg str out:       One of the modes accepted by fc._returner.

        :return:            The thresholded cube.
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.

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
        Plane-parallel raytracer. This routine merely solves the radiative transfer
        integral from the back to the front of the box. We assume that:

        0. That we are integrating along axis=0.
        1. The box size is 1
        2. cell width are uniform and are a/ni
        3. there is no background source
        4. the emissivity is proportional to the cube variable
        5. the absorption coefficient is proportional to the cube variable

        This raytracer is merely implemented for visualization purposes, which
        is the reason we have assumptions 4 and 5.

        :arg obj fc:                :class:`pyFC.FractalCube` object.
        :arg flt absorb_coeff:      Absorption coefficient   
        :arg flt emiss_coeff:       Emission coefficient   
        :arg int ax:                Direction of plane normal. {0|1|2}, 
                                    default is 0 ("y-z plane")

        :return:                    Raytraced image
        :rtype:                     2D ndarray

        """

        # Emission and absorption coefficients
        c_abs = absorb_coeff * fc.cube
        c_emi = emiss_coeff * fc.cube

        # Transmittance from one side of face to each cell
        # Note: c_abs*delta gives optical depths
        # Also make sure transmittance does not include current layer, so
        # the array needs to be shifted by one layer toward the far boundary
        # The near boundary layer should be filled with a plane of 1s.
        total_length = 1.
        if ax == 0: delta = total_length / fc.ni
        if ax == 1: delta = total_length / fc.nj
        if ax == 2: delta = total_length / fc.nk

        transmittance = np.cumprod(np.exp(-c_abs * delta), axis=ax)
        if ax == 0: transmittance = np.append(np.ones((1, fc.nj, fc.nk)),
                                              transmittance[:-1, :, :], axis=ax)
        if ax == 1: transmittance = np.append(np.ones((fc.ni, 1, fc.nk)),
                                              transmittance[:, :-1, :], axis=ax)
        if ax == 2: transmittance = np.append(np.ones((fc.ni, fc.nj, 1)),
                                              transmittance[:, :, :-1], axis=ax)

        # The resulting integrated result as a 2D array
        return np.sum(c_emi * (1. - np.exp(-c_abs * delta)) * transmittance, axis=0)


class FCDataEditor():
    def __init__(self):
        pass

    def mult(self, fc, factor, out='copy'):
        """
        Multiply the cube with a value.

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg flt factor:    Factor to multiply cube with
        :arg str out:       One of the modes accepted by fc._returner.

        :return:            Multiplied cube
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.
        """
        result = fc.cube * factor
        return fc._returner(result, out)

    def pow(self, fc, power, out='copy'):
        """
        Exponentiate the cube with a value.

        :arg obj fc:        :class:`pyFC.FractalCube` object.
        :arg flt power:     Power to exponentiate cube with
        :arg str out:       One of the modes accepted by fc._returner.

        :return:            Exponentiated cube
        :rtype:             Type depends on 'out'. By default it returns a copy
                            of the :class:`pyFC.FractalCube` object.

        """
 
        result = (fc.cube) ** power
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

    .. note:: Untested.
    """

    def __init__(self):

        import scipy.stats as sps

        self.__dict__.update(locals());
        del self.__dict__['self']

    def mean(self, fc, out='value'):
        """
        Calculate mean of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Mean of cube PDF
        :rtype:         Double

        """
        mean = np.mean(fc.cube)
        return self._stats_returner(fc, out, mean=mean)

    def std(self, fc, out='value'):
        """
        Calculate standard deviation of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Standard deviation of cube PDF
        :rtype:         Double

        """
        std = np.std(fc.cube)
        return self._stats_returner(fc, out, std=std)

    def var(self, fc, out='value'):
        """
        Calculate variance of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Variance of cube PDF
        :rtype:         Double

        """
        var = np.var(fc.cube)
        return self._stats_returner(fc, out, var=var)

    def median(self, fc, out='value'):
        """
        Calculate median of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Median of cube PDF
        :rtype:         Double

        """
        median = np.median(fc.cube)
        return self._stats_returner(fc, out, median=median)

    def rms(self, fc, out='value'):
        """
        Calculate root mean square of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        RMS of cube PDF
        :rtype:         Double

        """
        rms = np.sqrt(np.mean(fc.cube ** 2))
        return self._stats_returner(fc, out, rms=rms)

    def skew(self, fc, out='value'):
        """
        Calculate skew of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Skew of cube PDF
        :rtype:         Double

        """

        # TODO program without scipy
        skew = sps.skew(fc.cube)

        return self._stats_returner(fc, out, skew=skew)

    def kurt(self, fc, out='value'):
        """
        Calculate kurtosis of fractal cube PDF

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Kurtosis of cube PDF
        :rtype:         Double

        """

        # TODO program without scipy
        kurt = sps.kurtosis(fc.cube)

        return self._stats_returner(fc, out, kurt=kurt)

    def flatness(self, fc, out='value'):
        """
        Calculate flatness of fractal cube PDF.
        See appendix in Sutherland & Bicknell (2007).

        :arg obj fc:    Fractal cube object
        :arg str out:   Output mode. Usually 'value'.

        :return:        Flatness of cube PDF
        :rtype:         Double

        """

        # TODO implement this
        flat = 0

        return self._stats_returner(fc, out, flat=flat)

    def _stats_returner(self, fc, out, **kwargs):
        """
        Different output depending on ``out``.

        :arg str out:   Output mode. {'inplace'|'copy'|'value'|'values'}

                        - 'inplace': Sets the value of the stat attribute and returns cube.
                        - 'copy': Copy of fractal cube with calculated attribute value set.
                        - 'value': Return only the value
                        - 'values': No clue what this was supposed to be.

        :returns:       Result
        :rtype:         Depends on out. (See above.)

        :raise:         ValueError if unknown out.
        """


        if out == 'inplace':
            for k, v in kwargs: setattr(fc, k, v)
            return fc

        elif out == 'copy':
            copy = fc.copy()
            for k, v in kwargs: setattr(copy, k, v)
            return copy

        elif out == 'value':
            return kwargs.values()[0]

        elif out == 'values':
            return kwarg.values()

        else:
            print('Invalid output mode')
            raise (ValueError)


class FractalCube():
    """
    Base class for fractal cubes. This class contains basic reading, writing functions, copy function, and functions to
    calculate power spectra. The statistical parameters of the fractal cube are contained as members.
    """

    def __init__(self, ni=64, nj=64, nk=64,
                 kmin=1, mean=1, sigma=np.sqrt(5.), beta=-5./3.):

        """
        :arg int i,j,k:     number of points in :math:`i`, :math:`j`, :math:`k` directions.
        :arg int kmin:      :math:`k_\mathrm{min}` of cube.
        :arg flt mean:      the mean of the single point distribution.
        :arg flt sigma:     the variance of the single point distribution.
        :arg flt beta:      the power of :math:`D(k)`, the power spectrum.

        :var cube:          The fractal cube data itself.
        :var ndim:          dimensionality of cube.
        :var shape:         [ni, nj, nk]

        """

        self.__dict__.update(locals());
        del self.__dict__['self']

        # Save some useful values
        self.shape = [ni, nj, nk]
        self.ndim = np.sum(np.greater(self.shape, 1))

        # Dictionary of manipulators (just for bookkeeping)
        # {name of object: [class, functions...]}
        self.manipulators = OrderedDict((
            ['fc_slicer', [FCSlicer, 'slice', 'tri_slice']],
            ['fc_affine', [FCAffine, 'translate', 'permute', 'mirror']],
            ['fc_extractor', [FCExtractor, 'extract_feature', 'lthreshold']],
            ['fc_raytracer', [FCRayTracer, 'pp_raytrace']]
        ))
        self._get_all_manipulators()

        # Obtain box ratios
        norder = np.argsort(self.shape)
        inorder = np.argsort(norder)
        shp_ord = np.array(self.shape)[norder]
        self.cube_ratios = (1.0*shp_ord/shp_ord[2])[inorder]

        # Obtain maximum values of kmin for box directions,
        # which are smaller than the largest box dimension.
        # This is not used in cloud generation procedure.
        # Instead, the sampling vector is scaled appropriately.
        # However, kmin !< 1 in any diretion after the scaling,
        # and we check this here.
        self.cube_kmins = ckm = kmin*self.cube_ratios
        if np.any(np.less(ckm, 1.)):
            print('Error: kmin for at least one dimension is less than 1.')
            print('Increase the number of cells in that dimension, or increase kmin.')
            raise(ValueError)


    def copy(self):
        """
        Make a deep copy of this object

        :return:    A deep copy of this object
        :rtype:     :class:`pyFC.FractalCube`
        """

        return copy.deepcopy(self)

    def read_cube(self, fname='data.dbl', prec='double'):
        """
        Read cube from file. The data of the cube returned and also
        saved to the :attr:`pyFC.LogNormalFractalCube.cube' attribute

        :arg str fname:     Data filename
        :arg str prec:      Floating point precision of output {'double'|'single'}

        :return:            Data of cube read in
        :rtype:             ndarray

        :raise:             ValueError if prec is invalid
        """
        if prec == 'double':
            dtype = np.dtype('<f8')
        elif prec == 'single':
            dtype = np.dtype('<f4')
        else:
            ValueError('Unknown prec ' + prec)

        cube = np.fromfile(fname, dtype=dtype, count=-1)
        cube = cube.reshape(*self.shape)
        return self._returner(cube, 'ndarray')

    def write_cube(self, fname='data.dbl', app=True, pad=False, prec='double'):
        """
        Writes out a fractal cube data file in little endian, 
        double precision. Care is taken not to overwrite existing files.

        :arg str fname:     Data filename
        :arg bool app:      automatically append kmin, ni, nj, and nk values to filename.
                            If app == True, append kmin, ni, nj, nk values to filename.
                            If suffixed files '<base>-[0-9][0-9]<ext>' exist, appended
                            by a suffix one larger than the highest number found.
        :arg bool pad:      Pad the numbering with 0s
        :arg str prec:      Floating point precision of output {'double'|'single'}

        :return:            1 for successful write
        :rtype:             int

        :raise:             ValueError if prec is invalid
        """

        if prec == 'double':
            out = self.cube.astype('<f8')
        elif prec == 'single':
            out = self.cube.astype('<f4')
        else:
            ValueError('Unknown prec ' + prec)

        if app:
            ext = osp.splitext(fname)[1]
            base = osp.splitext(fname)[0]

            if pad:
                fname = (base + '_' + format(self.kmin, '0>2d') + '_' +
                         format(self.ni, '0>4d') +
                         ('x' + format(self.nj, '0>4d') if self.ndim > 1 else '') +
                         ('x' + format(self.nk, '0>4d') if self.ndim > 2 else '') +
                         ext)
            else:
                fname = (base + '_' + str(self.kmin) + '_' + format(self.ni) +
                         ('x' + str(self.nj) if self.ndim > 1 else '') +
                         ('x' + str(self.nk) if self.ndim > 2 else '') +
                         ext)

        fname = pt.unique_fname(fname, '-', '[0-9][0-9]')

        out.tofile(fname)
        return 1

    def _returner(self, result, out):
        """ 
        Given a result, return the correct thing given an "out mode"

        :arg ndarray result:    The FractalCube.cube result that should be returned, copied, etc.
        :arg str out:           The "out mode". Possible values are
                                'inplace':  save the ndarray data to self.cube return the FractalCube instance;
                                'ndarray':  save the ndarray data to self.cube and return the ndarray;
                                'copy':     return a copy of the FractalCube instance, with the ndarray data saved to the
                                copy's self.cube.

        :return:                Varies according to the "out mode", ``out``
        :rtype:                 Varies according to the "out mode", ``out``

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
            raise (ValueError)

    def _get_all_manipulators(self):
        """
        Create methods in self from self.manipulators. The manipulator objects
        are also created and stored in self.
        """

        # Create method attributes. 
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
        Power spectrum of a fractal cube data

        :arg ndarray cube:  FractalCube.cube data array

        :return:            Power spectrum of this fractal cube
        :rtype:             ndarray
        """
        if cube is None: cube = self.cube
        return self._power_spec(fft.fftn(cube))

    def iso_power_spec(self, kmag=None, cube=None, stdev=False):
        """
        Isotropic power spectrum of this fractal cube.

        :arg ndarray kmag:  Array of wave-vector magnitude values (same size and dimensions as cube)
        :arg ndarray cube:  Array with cube data
        :arg bool stdev:    Calculate standard deviation

        :return:            Isotropic power spectrum of this fractal cube. :math:`(D(|k|), |k|)`
        :rtype:             tup
        """
        if kmag is None: kmag = self.kmag_sampling()
        if cube is None: cube = self.cube
        return self._iso_power_spec(kmag, self._power_spec(fft.fftn(cube)),
                                    stdev)

    def kmag_sampling(self, ni=None, nj=None, nk=None):
        """
        Sampling space, :math:`|k|_{i,j,k}`. See definition of FFT, :math:`k` and frequency
        at http://docs.scipy.org/doc/numpy/reference/routines.fft.html

        :arg int ni, nj, nk:    Number of cells (sample points) in i, j, k directions, respectively

        :return:                3-D array with sampling vector magnitudes
        :rtype:                 ndarray
        """

        if ni is None: ni = self.ni
        if nj is None: nj = self.nj
        if nk is None: nk = self.nk

        sampli, samplj, samplk = fft.fftfreq(ni), fft.fftfreq(nj), fft.fftfreq(nk)
        k1di = sampli * ni / self.cube_ratios[0]
        k1dj = samplj * nj / self.cube_ratios[1]
        k1dk = samplk * nk / self.cube_ratios[2]
        ksqri, ksqrj, ksqrk = k1di * k1di, k1dj * k1dj, k1dk * k1dk
        kmag = np.sqrt(np.add.outer(np.add.outer(ksqri, ksqrj), ksqrk))

        return kmag

    def norm_spec(self, spectrum):
        """
        Normalize a spectrum so that apodizing 
        a random Gaussian cube with the root of spectrum
        does not change the mean and variance of the Gaussian

        :arg ndarray spectrum:  A 1-, 2-, or 3-dimensional spectrum

        :return:                Normalized spectrum, same shape as input spectrum
        :rtype:                 ndarray
        """

        shape, N = spectrum.shape, spectrum.size
        spectrum = np.ravel(spectrum)
        csqr = (N - 1.) / np.sum(np.real(spectrum[1:]))
        spectrum *= csqr;
        spectrum[0] = 1
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
        first element represents k0 + 1. If not, ravel and remove first elements
        of both kr and pspec.
        """

        if not raveled:
            kr = np.ravel(kr)
            pspec_r = np.ravel(pspec_r)
            bins = np.append(np.unique(kr), kr.max() + 1)

        psc, bins = np.histogram(kr, bins)
        psw, bins = np.histogram(kr, bins, weights=pspec_r)
        means = psw / psc

        binc = 0.5 * (bins[:-1] + bins[1:])

        if stdev or digitize:
            psd = np.digitize(kr, bins[1:-1])

        if stdev:
            pss, bins = np.histogram(kr, bins, weights=(pspec_r - means[psd]) ** 2)
            stdvs = np.sqrt(pss / psc)

            if digitize:
                return means, binc, stdvs, psd.reshape(self.shape)
            else:
                return means, binc, stdvs

        else:
            if digitize:
                return means, binc, psd.reshape(self.shape)
            else:
                return means, binc

    def _power_spec(self, F, k=None):
        """
        Power spectrum. Without the factor of 4*pi.
        """
        return np.real(np.conjugate(F) * F)

    def print_stats(self):
        """
        Prints stats of the Fractal cube, including (cubetype, ni, nj, nk, kmin, mean, sigma, beta)

        :return:    1 at completion of routine
        :rtype:     int
        """

        # TODO Finish programming this
        param = [self.cubetype, self.ni, self.nj, self.nk, self.kmin, self.mean, self.sigma, self.beta]
        param_str = locals().keys()[1:]

        for i in zip(param_str, param):
            print(param_str+str(param))

        return 1


class GaussianFractalCube(FractalCube):
    """
    Class for lognormal fractal cubes, as generated by taking a random gaussian field and
    apodizing its spectrum in fourier space with a power law.

    """
    # TODO refactor mean to be mu

    def __init__(self, ni=64, nj=64, nk=64, kmin=1,
                 mean=0, sigma=200., beta=-1.6666666666666):
        """
        :arg int ni,nj,nk:  number of points in :math:`i`, :math:`j`, :math:`k` directions
        :arg int kmin:      :math:`k_\mathrm{min}` of fractal cube
        :arg flt mean:      the mean of the lognormal PDF
        :arg flt sigma:     the standard deviation of the gaussian PDF
        :arg flt beta:      the power-law index of :math:`D(k)`, the power spectrum.

        The above parameters are all saved as attributes of an instance of this class.
        """

        #: (*str*) -- Cube type {'lognormal', 'gaussian'}.
        self.cubetype = 'gaussian'

        #: (*ndarray*) -- Gaussian fractal cube data as ndarray of shape :attr:`pyFC.GaussianFractalCube.shape`
        self.cube = None

        #: (*int*) -- Number of dimensions of the fractal cube data array :attr:`pyFC.GaussianFractalCube.cube`
        self.ndim = None

        #: (*tup*) -- Shape of the fractal cube data array :attr:`pyFC.GaussianFractalCube.cube`
        self.shape = None

        FractalCube.__init__(self, ni=ni, nj=nj, nk=nk, kmin=kmin, mean=mean, sigma=sigma, beta=beta)

    def func_target_spec(self, k, kmin=None, beta=None):
        """
        The target spectrum function

        :arg ndarray k:     A domain in :math:`k`-space to evaluate the target spectrum over.
        :arg flt kmin:      A lower cutoff wavenumber, :math:`k_\mathrm{min}`.
        :arg flt beta:      The power law index of the target spectrum, :math:`\beta`.

        :return:            Target spectrum. Same shape as k.
        :rtype:             ndarray
        """

        if kmin is None: kmin = self.kmin
        if beta is None: beta = self.beta

        kp = np.where(k > kmin, k, np.inf)
        return kp ** (beta - 2.)

    def gen_cube(self, history=False):
        """
        Generate a Gaussian fractal cube with the statistics saved as attributes in an instance of this object.

        :arg bool history:  If history is True, this routine returns the initial cube too.

        :return:            Gaussian fractal cube
        :rtype:             :class:`pyFC.GaussianFractalCube`
        """

        # Gaussian random field, and the corresponding lognormal random field
        grf = np.random.normal(self.mean, self.sigma, self.shape)

        # Get the ndim kmag array (isotropic k)
        kmag = self.kmag_sampling()

        # The target spectrum
        target_spec = self.func_target_spec(kmag)
        target_spec = self.norm_spec(target_spec)

        # The apodization values
        apod = np.sqrt(target_spec)

        # N-dimensional (3-dimensional) FFT log-normal cube
        Fg = fft.fftn(grf)

        # Apodize with power law
        Fga = Fg * apod

        # Fourier transform back (the imaginary part is negligible)
        cube = np.real(fft.ifftn(Fga))

        # Return cubes. In "history mode", the cubes in the 
        # beginning is also returned. The history mode is only
        # used through the routine history_cubes
        if history:
            return self._returner(grf, 'copy'), self._returner(cube, 'inplace')
        else:
            return self._returner(cube, 'inplace')

    def lnfc(self):
        """
        Corresponding Lognormal field.

        :return:    The corresponding Lognormal field.
        :rtype:     :class:`pyFC.LogNormalFractalCube` object
        """
        ln = mt.LogNormalPDF(self.mean, self.sigma, gstat=True)
        lnfc = LogNormalFractalCube(self.ni, self.nj, self.nk, ln.mu,
                                    ln.sigma, self.beta)
        lnfc.cube = np.exp(self.cube)
        return lnfc


class LogNormalFractalCube(FractalCube):
    """
    Class for lognormal fractal cubes, as generated by the method of
    `Lewis & Austin (2002) <https://ams.confex.com/ams/11AR11CP/techprogram/paper_42772.htm>`_.
    This class contains basic reading, writing, generating functions, copy function, and functions to
    calculate power spectra. Functions to calculate the power spectra exist because they are needed in the
    actual construction routine. The statistical parameters of the fractal cube are contained as attributes.
    """
    # TODO refactor mean to be mu

    def __init__(self, ni=64, nj=64, nk=64, kmin=1,
                 mean=1, sigma=np.sqrt(5.), beta=-1.66666666666):
        """
        :arg int ni,nj,nk:  number of points in :math:`i`, :math:`j`, :math:`k` directions
        :arg int kmin:      :math:`k_\mathrm{min}` of fractal cube
        :arg flt mean:      the mean of the lognormal PDF
        :arg flt sigma:     the standard deviation of the lognormal PDF
        :arg flt beta:      the power-law index of :math:`D(k)`, the power spectrum.

        The above parameters are all saved as attributes of an instance of this class.
        """

        # Documented members and their defaults.

        #: (*str*) -- Cube type {'lognormal', 'gaussian'}.
        self.cubetype = 'lognormal'

        #: (*ndarray*) -- The lognormal fractal cube data as an arrayof shape :attr:`pyFC.LogNormalFractalCube.shape`
        self.cube = None

        #: (*int*) -- Number of dimensions of the fractal cube data array :attr:`pyFC.LogNormalFractalCube.cube`
        self.ndim = None

        #: (*tup*) -- Shape of the fractal cube data array :attr:`pyFC.LogNormalFractalCube.cube`
        self.shape = None

        FractalCube.__init__(self, ni=ni, nj=nj, nk=nk, kmin=kmin, mean=mean, sigma=sigma, beta=beta)


    def func_target_spec(self, k, kmin=None, beta=None):
        """
        The target spectrum function

        :arg ndarray k:     A domain in :math:`k`-space to evaluate the target spectrum over.
        :arg flt kmin:      A lower cutoff wavenumber, :math:`k_\mathrm{min}`.
        :arg flt beta:      The power law index of the target spectrum, :math:`\beta`.

        :return:            Target spectrum. Same shape as k.
        :rtype:             ndarray
        """

        if kmin is None: kmin = self.kmin
        if beta is None: beta = self.beta

        kp = np.where(np.greater_equal(k, kmin), k, np.inf)
        return kp ** (beta - 2.)

    def gen_cube(self, verbose=True):
        """
        Generate a lognormal fractal cube by the method of
        `Lewis & Austin (2002) <https://ams.confex.com/ams/11AR11CP/techprogram/paper_42772.htm>`_
        with the statistical properties saved in the instance of the :class:`pyFC.FractalCube` object.

        :arg bool verbose:  Verbosity switch.

        :return:            The generated lognormal fractal cube
        :rtype:             :class:`pyFC.LogNormalFractalCube` object

        .. note:: The actual code for generating fractal cubes is in :meth:`pyFC.LogNormalFractalCube.yield_cubes`.
        """

        for fc in self.yield_cubes(history=False, verbose=verbose):
            return fc

    def yield_cubes(self, history=True, verbose=True):
        """
        Generate a lognormal fractal cube by the method of
        `Lewis & Austin (2002) <https://ams.confex.com/ams/11AR11CP/techprogram/paper_42772.htm>`_
        with the statistical properties saved in the instance of the FractalCube object.

        :arg bool verbose:  Verbosity switch.
        :arg bool history:  History switch. If true, generate a cube for each iteration.
                            This parameter should always be true, unless this function is called from
                            :meth:`pyFC.LogNormalFractalCube.gen_cube`.

        :return:            Yields lognormal fractal cubes
        :rtype:             :class:`pyFC.LogNormalFractalCube` object

        If history is True, this routine yields the cubes for each iteration.
        This option is actually not to be used. Use gen_cube instead.
        """

        #_____________________________________
        # Parameters (These can be changed)

        # Iteration 
        self.iter_max = 10
        self.iter_tol = 0.01

        # Correction factor
        self.eta = 0.5

        #_____________________________________
        # Main bit

        # Theoretical lognormal object
        ln = mt.LogNormalPDF(self.mean, self.sigma)

        # Convert to gaussian stats
        mean_g, sigma_g = ln.mu_g, ln.sigma_g

        # Gaussian random field, and the corresponding lognormal random field
        grf = np.random.normal(mean_g, sigma_g, self.shape)

        # Yield cube, if history is on
        if history: yield self._returner(np.exp(grf), 'copy')

        # Get the ndim kmag array (isotropic k)
        kmag = self.kmag_sampling(self.ni, self.nj, self.nk)

        # Isotropic k (kmag is also used as a dummy second argument)
        dummy, k_iso = self._iso_power_spec(kmag, kmag)

        # Some helpful arrays for log k
        lkmag, lk_iso, lkmin = mt.zero_log10(kmag), np.log10(k_iso), np.log10(self.kmin)

        # Some helpful indices that determine where fits and 
        # corrections are applied an. These could have been defined in
        # linear space, but this doesn't change the indexing. The important
        # thing is that there is one for k_iso and one for kmag. sf refers to
        # indices for fitting region, so, for the other region. 
        sf_lk_iso = np.s_[lk_iso >= lkmin]
        so_lk_iso = np.s_[np.logical_not(sf_lk_iso)];
        so_lk_iso[0] = False
        sf_lkmag = np.s_[lkmag >= lkmin]
        so_lkmag = np.ravel(np.s_[np.logical_not(sf_lkmag)]);
        so_lkmag[0] = False;
        so_lkmag.reshape(self.shape)

        # The target spectrum
        target_spec = self.func_target_spec(kmag)
        target_spec = self.norm_spec(target_spec)
        target_spec_iso = self.func_target_spec(k_iso)

        # The apodization values
        apod = np.sqrt(target_spec)

        # N-dimensional (3-dimensional) FFT lognormal cube
        Fg = fft.fftn(grf)

        # Apodize with power law
        Fga = Fg * apod

        # Loop begins here
        convergence, iiter = 1, 1
        while convergence > self.iter_tol and iiter <= self.iter_max:

            # Fourier transform back (the imaginary part is negligible)
            grf_a = np.real(fft.ifftn(Fga))

            # Create lognormal
            lrf_a = np.exp(grf_a)

            # Yield cube, if history
            if history: yield self._returner(lrf_a, 'copy')

            # Power spectrum of lognormal is not desired power-law
            Fla = fft.fftn(lrf_a)
            Dla = self._power_spec(Fla)

            # Isotropic power spectra
            Dla_iso, k_iso = self._iso_power_spec(kmag, Dla)

            # Fits to the isotropic spectra

            # zeroth order fit, to find best height for target spectrum 
            # (kind of normalization). The multiplication by 10**fit0
            # retains zeroes in log space.
            weights = np.r_[np.diff(k_iso), np.diff(k_iso[-2:])]
            fit0 = np.average(mt.zero_log10(Dla_iso[sf_lk_iso]) -
                              mt.zero_log10(target_spec_iso[sf_lk_iso]),
                              weights=weights[sf_lk_iso])
            #p0_iso = mt.zero_log10(10 ** fit0 * target_spec_iso)

            # Fit power spec of lognormal with polynomials
            fit2 = np.polyfit(lk_iso[sf_lk_iso], mt.zero_log10(Dla_iso[sf_lk_iso]), 2)
            #p2_iso = np.polyval(fit2, lk_iso)

            # Corrections based on fits. fit0 needs to be multiplied
            # with the func_target_spec, otherwise 0s are not preserved.
            p2 = np.polyval(fit2, lkmag)
            p0 = mt.zero_log10(10 ** fit0 * self.func_target_spec(kmag))
            corr = np.where(sf_lkmag, self.eta * (p0 - p2), 0)

            # Apply correction (in log space) to apodization spectrum
            corr_apod = apod * 10 ** corr

            # Re-Apodize with normalized power law
            # From here, it's the same as before the loop.
            corr_apod2 = self.norm_spec(corr_apod ** 2)
            apod_old, apod = apod.copy(), np.sqrt(corr_apod2)
            Fga = Fg * apod


            # Estimate convergence
            convergence = np.average(mt.zero_div(abs(self._power_spec(apod_old) -
                                                     self._power_spec(apod)),
                                                 self._power_spec(apod)))
            # Some messages
            if verbose:
                print('iteration ' + str(iiter))
                print('convergence = ' + str(convergence))
                print('')

            # Get ready for next iteration
            iiter += 1

        # A last conversion

        # Fourier transform back (the imaginary part is negligible)
        grf_a = np.real(fft.ifftn(Fga))

        # Create lognormal
        cube = np.exp(grf_a)

        # Return cube object
        yield self._returner(cube, 'inplace')

    def gnfc(self):
        """
        The corresponding Gaussian field.

        :return:    The corresponding Gaussian field.
        :rtype:     :class:`pyFC.GaussianFractalCube` object
        """
        ln = mt.LogNormalPDF(self.mean, self.sigma)
        gnfc = GaussianFractalCube(self.ni, self.nj, self.nk, self.kmin, ln.mu_g,
                                   ln.sigma_g, self.beta)
        gnfc.cube = np.log(self.cube)
        return gnfc


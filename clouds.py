import numpy as np
import parsetools as pt
from mathtools import LogNormalPDF
from collections import OrderedDict
import glob

class FractalCubeData():
    """
    members
    -------
    cube             The fractal cube data itself, in (res[0], res[1], res[2])
    res              [res0, res1[, res2]] array with resolution of each dimension
    kim              kmin of cube
    dim              dim of cube
    file_name        filename of file to extract cube from

    Note: the data is not necessarily a "cube". We just retained this word as most
    of our fractal cloud data was generated/used as a cube.
    """

    def __init__(self, fname='rho.dbl', nx=1, ny=1, nz=1, kmin=1, sigma=5., alpha=-1.66666666666):
        """
        Fractal cube class. Upon creation, it either reads a cube from a file of
        name fname, if it exists, or creates a cube with given parameters, if
        the file doesn't exist yet. (A file can be written by invoking
        write_cube().) In the former case, the parameters should also be given as they are not
        contained in the cube file.

        arguments
        ---------
        fname       filename of file to extract cube from
        n{x|y|z}    number of points in x, y, z directions
        kmin        kmin of cube
        sigma       the variance of the lognormal density distribution
        alpha       the power of the D(k), the density power spectrum.

        members
        -------
        cube         
        dim         dimensionality of cube
        zero_index  the index of the first cube cell. This is to support a
                    a global index system, and treat subcubes as FractalCube
                    objects.
        fname_num   number pattern to avoid filename overwrite
        fname_ext   extension of filename
        fname_base  basename of filename
        
        About the cubes
        ---------------
        The mean of the lognormal density distribution is assumed to be 1. kmin
        is the Nyquist limit N/2-1

        """

        self.__dict__.update(locals()); del self.__dict__['self']

        self.res = [nx, ny, nz]
        self.dim = np.sum(np.greater(self.res, 1))
        self.zero_index = (0,0,0)

        if glob.glob(fname): self.cube = self._read_cube()
        else: self.cube = self._gen_cube() ## Not yet programmed!

    def _read_cube(self):
        cube = np.fromfile(self.fname, dtype=np.dtype('<f8'), count=-1)
        return cube.reshape(*self.res)

    def _gen_cube(self):
        """
        Generate a fractal cube with the given statistics and save it to file
        self.fname

        A faire
        """

    def write_cube(self, fname=None):
        """
        This routine writes out a fractal cube data file in little endian, 
        double precision. Care is taken not to overwrite existing files.
        """

        ## If no fname, use self.fname. If suffixed files '<base>-[0-9][0-9]<ext>' 
        ## exist, appended by a suffix one larger than the highest number found.
        if fname == None: fname = self.fname
        fname = pt.unique_fname(fname, '-', '[0-9][0-9]')

        self.cube.tofile(fname)
        return 1

    def copy(self):
        """ Return a copy of this object """

        return FractalCube(self.fname, self.nx, self.ny, self.nz, 
                           self.kmin, self.sigma, self.alpha)

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
            return result
        elif out == 'copy':
            copy = self.copy()
            copy.cube = result
            return copy
        else:
            print('Invalid output mode')
            raise(ValueError)

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

    def mirror(self, fc, ax=0, out='inplace'):
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
                        bgv=1.e-3, trim=False, out='copy'):
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

        A faire
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

    def pp_raytrace(self, fc, absorb_coeff=1., emiss_coeff=1.):
        """
        Plane-parallel raytracer.

        Returns a 2D ndarray of a raytraced image given absorption and emission
        coefficients. This routine merely solves the radiative transfer
        integral from the back to the front of the box. We assume that:

            0. That we are integrating along axis=0.
            1. The box size is 1
            2. cell width are uniform and are a/nx
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
        delta = total_length/fc.nx
        transmittance = np.cumprod(np.exp(-c_abs*delta), axis=0)
        transmittance = np.append(np.ones((1,fc.ny,fc.nz)), 
                                  transmittance[:-1,:,:], axis=0)

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

class FractalCube(FractalCubeData):

    def __init__(self, fname='rho.dbl', 
                 nx=1, ny=1, nz=1, 
                 kmin=1, sigma=5., alpha=-1.66666666666):

        self.__dict__.update(locals()); del self.__dict__['self']

        ## Init all classes
        FractalCubeData.__init__(self, fname=fname, 
                                 nx=nx, ny=ny, nz=nz, 
                                 kmin=kmin, sigma=sigma, alpha=alpha)

        ## Dictionary of manipulators (just for bookkeeping)
        ## {name of object: [class, functions...]}
        self.manipulators = OrderedDict((
            ['fc_slicer',    [FCSlicer,    'slice', 'tri_slice']],
            ['fc_affine',    [FCAffine,    'translate', 'permute', 'mirror']],
            ['fc_extractor', [FCExtractor, 'extract_feature', 'lthreshold']],
            ['fc_raytracer', [FCRayTracer, 'pp_raytrace']]
        ))

        self._get_all_manipulators()

    def _get_all_manipulators(self):
        """
        Create methods in self from self.manipulators. The manipulator objects
        are also created and stored in self.
        """

        ## Create method attributes. 
        for o, m in self.manipulators.items():
            for f in m[1:]: setattr(self, f, self._m_curried(getattr(m[0](), f)))

    def _m_curried(self,m_func):
        """
        Currying function for creating method attributes whose first argument is
        the fractal cube, self, itself, rather an external cube instance.
        """
        def m_curry(*args, **kwargs): return m_func(self, *args, **kwargs)
        return m_curry



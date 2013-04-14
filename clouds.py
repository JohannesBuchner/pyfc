import numpy as np

class FractalCube():
    """
    members
    -------
    cube             The fractal cube data itself, in (res[0], res[1], res[2])
    res              [res0, res1[, res2]] array with resolution of each dimension
    kim              kmin of cube
    dim              dim of cube
    file_name        filename of file to extract cube from

    Note: the data is not necessarily a cube. We just retained this word as most
    of our fractal cloud data was generated/used as a cube.
    """

    def __init__(self, fname='rho.dbl', nx=1, ny=1, nz=1, kmin=1, sigma=5., alpha=-1.66666666666):
        """
        arguments
        ---------
        fname       filename of file to extract cube from
        n{x|y|z}    number of points in x, y, z directions
        kmin        kmin of cube
        sigma       the variance of the lognormal density distribution
        alpha       the power of the D(k), the density power spectrum.

        members
        -------
        zero_index  the index of the first cube cell. This is to support a
                    a global index system, and treat subcubes as FractalCube
                    objects.
        file_name   the file name of the 
        
        
        About the cubes
        ---------------
        The mean of the lognormal density distribution is assumed to be 1. kmin
        is the Nyquist limit N/2-1

        """

        self.__dict__.update(locals()); del self.__dict__['self']

        self.res = [nx, ny, nz]
        self.dim = np.sum(np.greater(self.res, 1))
        self.zero_index = (0,0,0)
        self.cube = self.read_cube()

    def read_cube(self):
        cube = np.fromfile(self.fname, dtype=np.dtype('<f8'), count=-1)
        return cube.reshape(*self.res)

    def gen_cube(self):
        """
        Generate a fractal cube with the given statistics and save it to file
        self.fname

        A faire
        """

    def copy():
        """ Return a copy of this object """
        return self.__init__(fname=self.fname, nx=self.nx, ny=self.ny1, nz=self.nz
                               kmin=self.kmin, sigma=self.sigma., alpha=self.alpha)

    def returner(result, out):
        """ 
        Given a result, return the correct thing given an out mode
        copy:    return a copy of the FractalCube object
        ndarray:     return array
        inplace: default, change in place
        """
        if out == 'inplace':
            self.cube = result
            return None
        elif out == 'ndarray':
            return result
        elif out == 'copy':
            return self.copy()
        else:
            print('Invalid output mode')
            raise(ValueError)

    def tri_slice(self, locs=(0.5,0.5,0.5), scale='frac', out='ndarray'):
        """ 
        Description
        -----------
        A generator function for three perpendicular slices at point loc

        Arguments
        ---------
        locs     location of tri_slice (point where planes intersect)
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.
        out      One of the modes accepted by self.returner, except
                 "inplace"

        Yields
        ------
        Slice arrays for each axis and intercept.
        This will always yield three slices, regardless 
        of whether cube is 2D
        """

        if out == 'inplace': 
            print('out="inplace" not allowed')
            raise(ValueError)

        if scale == 'frac': locs = np.round(locs*np.array(self.cube.shape))

        for i, loc in enumerate(locs): 
            result = np.rollaxis(self.cube, i)[loc,:,:]
            yield returner(result, out)

    def slice(self, ax=0, loc=0.5, scale='frac', out='ndarray'):
        """ 
        ax       axis perpendicular to slice plane
        loc      location of slice on that axis
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.
                 out      One of the modes accepted by self.returner, except
                          "inplace"

        Returns:   2D array of slice
        """

        if out == 'inplace': 
            print('out="inplace" not allowed')
            raise(ValueError)

        if scale == 'frac': loc = np.round(loc*self.cube.shape[ax])

        result = np.rollaxis(self.cube, ax)[loc,:,:]

        return returner(result, out)

    def translate(self, ax=0, delta=0.5, scale='frac', out='inplace'):
        """ 
        Translation with roll
        ax       axis along which to translate values
        delta    distance of translation
        scale    {'idx'|'frac'} integer index (idx) or fraction of cube width (rounded to
                 nearest integer, frac) for location. Default 'frac'.
        out      One of the modes accepted by self.returner

        Returns:   translated data
        """


        if scale == 'frac': delta = np.round(delta*self.cube.shape[ax])

        result = np.roll(self.cube, delta, axis=ax)
        return returner(result, out)

    def permute(self, choice=0, out='inplace'):
        """ 
        Chooses one of six permutations of the cube's axis orders
        choice   choses one of six permuations:
            0: 012, 1: 210 = 0.T, 
            2: 120, 3: 021 = 2.T, 
            4: 201, 5: 102 = 4.T, 
            where T is the transpose
        out      One of the modes accepted by self.returner

        Returns:   translated data
        """

        if choice%2 == 0: result = np.rollaxis(self.cube , choice%3)
        else:             result = np.rollaxis(self.cube , choice%3).T

        return returner(result, out)

    def mirror(self, ax=0, out='inplace'):
        """
        Mirrors a cube about the midplane along axis ax
        out      One of the modes accepted by self.returner
        """
        if ax == 0: result = self.cube[::-1,:,:]
        if ax == 1: result = self.cube[:,::-1,:]
        if ax == 2: result = self.cube[:,:,::-1]

        return returner(result, out)

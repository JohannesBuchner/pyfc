from clouds import FractalCube
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm



def _input2fc(input):
    """ 
    Creates and returns FractalCube instance if input is a text
    with filename. Returns FractalCube instance if input is one
    already.
    """
    if (isinstance(input, basestring) 
        if sys.version_info < (3,3) 
        else isinstance(input, str)):
        return FractalCube(input, nx, ny, nz, kmin, sigma, alpha)
    else: 
        return input

def translate(fc, ax=0, delta=0.5, scale='frac'):
    """
    See FractalCube.translate
    """
    return fc.translate(ax, delta, scale, out='copy')

def permute(fc, choice=0):
    """
    See FractalCube.permute with out='copy'
    """
    return fc.permute(choice, out='copy')

def mirror(fc, ax=0):
    """
    See FractalCube.mirror
    """
    return fc.mirror(ax, out='copy')

def plot_midplane_slice(input='rho.dbl', nx=512, ny=512, nz=512, 
                   ax=0, scaling='lin', cmap=cm.copper,
                   kmin=1, sigma=5., alpha=-1.666666666): 
    """
    Create and plot midplane slice
    file        filename
    nx, ny, nz  dimensions of fractal cube
    ax          direction of plane normal. {0|1|2}, default 0 ("y-z plane")
    scaling     linear or log data map
    cm          colormap. any colormap object (default cm.copper)
    kmin        statistical parameters of fractal cube
    sigma                 ||
    alpha                 ||
    """

    fc = _input2fc()

    pl.ion()

    if   scaling == 'lin': vmin = None; vmax = None; data = fc.slice(ax)
    elif scaling == 'log': vmin = -3;   vmax = 3;    data = np.log10(fc.slice(ax))

    pl.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

def plot_tri_midplane_slice(input='rho.dbl', nx=512, ny=512, nz=512, 
                       scaling='lin', cmap=cm.copper,
                       kmin=1, sigma=5., alpha=-1.666666666): 
    """
    Create and plot midplane slice
    file        filename
    nx, ny, nz  dimensions of fractal cube
    ax          direction of plane normal. {0|1|2}, default 0 ("y-z plane")
    scaling     linear or log data map
    cm          colormap. any colormap object (default cm.copper)
    kmin        statistical parameters of fractal cube
    sigma                 ||
    alpha                 ||
    """

    fc = _input2fc(input)

    pl.ion()

    if   scaling == 'lin': vmin = None; vmax = None; 
    elif scaling == 'log': vmin = -3;   vmax = 3; 

    for i, data in enumerate(fc.tri_slice()):
        pl.figure()
        if scaling == 'log': data = np.log10(data)
        pl.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)


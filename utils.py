from clouds import FCSlicer
from clouds import FCAffine
from clouds import FCExtractor
from clouds import FCRayTracer
from clouds import FractalCube
import sys
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import parsetools as pt

## write this up in Podio and evernote

## Map fundumental manipulation functions from clouds.py 
## to names here. The fundamental manipulations are
## listed in FractalCube.__init__
slice     = FCSlicer().slice
tri_slice = FCSlicer().tri_slice

translate = FCAffine().translate
permute   = FCAffine().permute
mirror    = FCAffine().mirror

extract_feature = FCExtractor().extract_feature
lthreshold      = FCExtractor().lthreshold

pp_raytrace = FCRayTracer().pp_raytrace


## Other functions
def _input2fc(cube, nx, ny, nz, kmin, sigma, alpha):
    """ 
    Creates and returns FractalCube instance if cube is a text
    with filename. Returns FractalCube instance if cube is one
    already.
    """
    if (isinstance(cube, basestring) if sys.version_info < (3,3) else isinstance(cube, str)):
        return FractalCube(cube, nx, ny, nz, kmin, sigma, alpha)
    else: 
        return cube

def write_cube(fc, fname=None): 
    """
    See FractalCube.write_cube
    """
    return fc.write_cube(fname)

def plot_midplane_slice(cube='rho.dbl', nx=512, ny=512, nz=512, 
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

    fc = _input2fc(cube, nx, ny, nz, kmin, sigma, alpha)
    fcs = FCSlicer()

    pl.ion()

    if   scaling == 'lin': vmin = None; vmax = None; data = fcs.slice(fc, ax)
    elif scaling == 'log': vmin = -3;   vmax = 3;    data = np.log10(fcs.slice(fc, ax))

    pl.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

def plot_tri_midplane_slice(cube='rho.dbl', nx=512, ny=512, nz=512,
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

    fc = _input2fc(cube, nx, ny, nz, kmin, sigma, alpha)
    fcs = FCSlicer()

    pl.ion()

    if   scaling == 'lin': vmin = 1.e-3; vmax = None; 
    elif scaling == 'log': vmin = 0; vmax = None; 

    fig, sax = pl.subplots(1, 3, squeeze=True, sharey=True, figsize=(9,3))
    pl.subplots_adjust(wspace=0.01)
    for i, data in enumerate(fcs.tri_slice(fc)):
        if scaling == 'log': data = np.log10(data)
        im = sax[i].imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

    #pl.colorbar(im)

def plot_raytrace(cube='rho.dbl', nx=512, ny=512, nz=512,
                  scaling='lin', cmap=cm.copper, 
                  absorb_coeff=1., emiss_coeff=1.,
                  vmin=None, vmax=None,
                  kmin=1, sigma=5., alpha=-1.666666666): 
    """
    Emissivity has same units as intensity (the ray integration quantity)
    Integration in axis=0 direction.
    Assume width of cube box is 1, so that deltax = 1/nx
    """

    fc = _input2fc(cube, nx, ny, nz, kmin, sigma, alpha)
    fcr = FCRayTracer()

    pl.ion()

    ## Some normed vmin and vmax are good for visualization
    #if   scaling == 'lin': vmin = None; vmax = None; 
    if   scaling == 'lin': vmin = 0.005; vmax = 10; 
    elif scaling == 'log': vmin = None; vmax = None; 

    image = fcr.pp_raytrace(fc, absorb_coeff, emiss_coeff)

    if scaling == 'log': image = np.log10(image)
    pl.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)

def flythrough_raytrace(cube='rho.dbl', nx=512, ny=512, nz=512,
                        scaling='lin', cmap=cm.copper, 
                        absorb_coeff=1., emiss_coeff=1.,
                        vmin=None, vmax=None,
                        kmin=1, sigma=5., alpha=-1.666666666,
                        fname='raytrace-00.png'):
    """
    Creates a raytrace cube through which one could fly.  
    Integration and rolling (translation) in axis=0 direction.

    This does not work with the above plane-parallel ray-trace
    Because there are not enough depth cues. Just brightening 
    is not enough. Need 3D perspective, rays from one point, a 
    certain distance away from the cube and then passing at 
    various angles through the cube. Need only one new parameter,
    distance from cube, but need to program random ray 
    intersections through cube.

    Use "<fname>-00" format for fname.
    """

    fc = _input2fc(cube, nx, ny, nz, kmin, sigma, alpha)
    fca = FCAffine()

    pl.figure()
    plot_raytrace(fc, absorb_coeff=absorb_coeff, emiss_coeff=emiss_coeff)
    pl.savefig(pt.unique_fname(fname))
    for i in range(fc.nx-1):

        ## Translate happens in place
        fca.translate(fc, delta=1, scale='lin')
        plot_raytrace(fc, absorb_coeff=absorb_coeff, emiss_coeff=emiss_coeff)
        pl.savefig(pt.unique_fname(fname))


## Create yielding version of FCExtractor.extract_feature and create a funciton
## here that plots the output


## Just for reference
#imshow(np.log10(np.ma.masked_array(fc.cube[:,:,32],
#np.not_equal(snm.label(np.where(np.less(fc.cube, 2.0), 0, fc.cube))[0], 4)[:,:,32])))


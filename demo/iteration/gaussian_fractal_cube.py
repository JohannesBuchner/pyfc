"""
_________________________________________________________________________________
Description

Create a log-normal fractal cube from a by the iterative method of 
Lewis & Austin (2002)

"""

import pyFC
import numpy as np
import matplotlib.pyplot as pl


"""
_________________________________________________________________________________
Input parameters

"""

## Dimensions and statistics of cube
## Although different ni, nj, and nk are allowed
## the respective kmin in each direction are not
## independently definable yet
ni, nj, nk = 64, 64, 64
mean, sigma = 0., 200.
beta, kmin =  -5./3., 1.

## Do plots? Save file?
## Get cubes from each iteration?
plot = True
save = False

## Filename
fname = 'rho.dbl'


"""
_________________________________________________________________________________
Main bit

"""

## Create Gaussian random field 
fc = pyFC.GaussianFractalCube(ni, nj, nk, kmin=kmin, 
                              mean=mean, sigma=sigma, beta=beta)

## Plot
if plot:

    pl.ion()

    ## Plot the cube
    fc.gen_cube()
    pyFC.plot_field_stats(fc, scaling='lin', vmin=-200, vmax=200)


## Just make cube
else:
    fc.gen_cube()

## Save output file
if save:
    fc.write_cube(fname)



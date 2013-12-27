"""
_________________________________________________________________________________
Description

Create a log-normal fractal cube from a by the iterative method of 
Lewis & Austin (2002)

"""

import pyFC
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import parsetools as pt


"""
_________________________________________________________________________________
Input parameters

"""

## Dimensions and statistics of cube
## Although different ni, nj, and nk are allowed
## the respective kmin in each direction are not
## independently definable yet
ni, nj, nk = 64, 64, 64
mean, sigma = 1., np.sqrt(5.)
beta, kmin =  -5./3., 1

## Do plots? Save file? Save plots?
## Get cubes from each iteration?
plot = True
save = False
saveplot = True
history = True
cmap = cm.jet
plottype = 'imshow'

## Cube and plot filenames
fname  = 'rho_n'+str(ni)+'_k'+str(int(kmin))+('-00' if history else '')+'.dbl'
pname1 = 'rho_n'+str(ni)+'_k'+str(int(kmin))+('-00' if history else '')+'.png'
pname2 = 'rho_n'+str(ni)+'_k'+str(int(kmin))+('-00' if history else '')+'.pdf'


"""
_________________________________________________________________________________
Main bit

"""

## Create Gaussian random field 
fc = pyFC.LogNormalFractalCube(ni, nj, nk, kmin=kmin, 
                               mean=mean, sigma=sigma, beta=beta)

## Plot
if plot:

    pl.ion()

    ## Plot each cube
    if history:
        for fci in fc.yield_cubes():
            pyFC.plot_field_stats(fci, scaling='log', cmap=cmap, 
                                  plottype=plottype, vmin=-1.9, vmax=1.9)
            if saveplot:
                if plottype == 'imshow':
                    pl.savefig(pt.unique_fname(pname1))
                    pl.savefig(pt.unique_fname(pname2))
                elif plottype == 'pcolormesh':
                    pl.savefig(pt.unique_fname(pname1))

    ## Or just the final one
    else:
        fc.gen_cube()
        pyFC.plot_field_stats(fc, scaling='log', cmap=cmap, 
                              plottype=plottype, vmin=-1.9, vmax=1.9)
        if saveplot:
            if plottype == 'imshow':
                pl.savefig(pt.unique_fname(pname1))
                pl.savefig(pt.unique_fname(pname2))
            elif plottype == 'pcolormesh':
                pl.savefig(pt.unique_fname(pname1))

## Just make cube
else:
    fc.gen_cube()

## Save output file
if save:
    fc.write_cube(fname)



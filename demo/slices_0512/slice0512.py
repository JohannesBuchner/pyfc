"""
________________________________________________________________
Description

This file just quickly creates some 2D cubes and plots slices 
to demonstrate different values of kmin for a 512^3 field.

"""
import pyFC as pfc
import matplotlib.cm as cm
import matplotlib.pyplot as pl

## kmins to use
kmins = [1., 2., 4., 8., 10., 12., 16., 20., 32., 40.]

## plottype {'pcolormesh'|'imshow'}
plottype = 'imshow'

## Loop over kmins
for kmin in kmins:

    ## Create fractal cubes
    fc = pfc.LogNormalFractalCube(512, 512, 1, kmin, 1.)
    fc.gen_cube()

    ## Plot slice. 
    pfc.plot_midplane_slice(fc, ax=2, kmin=kmin, scaling='log', cmap=cm.jet, 
                            colorbar=(True if kmin==kmins[-1] else False),
                            labels=False, plottype=plottype, kminlabel=True, 
                            vmin=-1.7, vmax=1.7)

    ## Save images. Only save png for pcolormesh.
    if plottype == 'imshow':
        pl.savefig('512_'+format(int(kmin), '>02')+'-im.png')
        pl.savefig('512_'+format(int(kmin), '>02')+'-im.pdf')
    elif plottype == 'pcolormesh':
        pl.savefig('512_'+format(int(kmin), '>02')+'-pm.png')
    else:
        ValueError('slice0512: Unknown plottype, '+plottype)


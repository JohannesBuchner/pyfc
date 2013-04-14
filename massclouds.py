#!/usr/bin/python
import matplotlib as mpl
import numpy as np
import matplotlib.pylab as pl



## ===============================================
## Cubes
## They can be found on /priv/myriad3/ralph/cubes
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#fname = "/priv/myriad3/ralph/cubes/k12km_512F64_3D_L00.dat"
#fname = "/home/ayw/research/clouds/cubes/k05km_512.dat"
#fname = "/home/ayw/research/clouds/cubes/k20km_512.dat"
fname = "k20km_512.dat"
#fname = "/priv/myriad3/ralph/cubes/k10km_512.dat"
#fname = "/priv/myriad3/ralph/cubes/k05km_512.dat"
#fname = "/priv/myriad3/ralph/cubes/k04km_512F64_L00.dat"
#fname = "/priv/myriad3/ralph/cubes/k03km_512F64_L00.dat"
#fname = "/priv/myriad3/ralph/cubes/k02km_512_cen.dat"
#fname = "/priv/myriad3/ralph/cubes/k01km_512_c3en.dat"

## ~~~~~~~~~~~~~~~~
## Cube properties
## ----------------

ndim = 3
res = 512


## =======================
## Some physical constants
## ~~~~~~~~~~~~~~~~~~~~~~~

## Mass of sun
Msun = 2.0e33

## kiloparsec
kpc = 3.086e21

## Boltzmann's constant 
kBolz = 1.380626e-16

## Mass of proton
mp = 1.67261411e-24

## One amu
amu = 1.660540210E-24

## Mean mass per particle
mu = 0.6156

## elementary charge in SI
ec = 1.602e-19


## ====================
## Analysis parameters
## ~~~~~~~~~~~~~~~~~~~~

## Some control over which
## parts to execute
mcalc = True
look = True

smooth = False
rapid = True


## ~~~~~~~
## Domain
## -------

## Mode in which to clip
## Can be cuboid or spherical
volume = 'cuboid'

## cube has coordinates such that 
## the extents are the following
cubesize = 10*kpc

cubexmin = 0*kpc
cubexmax = 10*kpc
cubeymin = -5*kpc
cubeymax = 5*kpc
cubezmin = -5*kpc
cubezmax = 5*kpc


## We want to work with cuboid 
## with following corners
xmin = 0*kpc
xmax = 5*kpc
ymin = -5*kpc
ymax = 5*kpc
zmin = -5*kpc
zmax = 5*kpc

## Cloud smoothing region
outer_rim_width = 1.5*kpc

## Slice location (for visualization)
slicepos = 0*kpc
sliceaxnorm = 'z'

## For spherical distribution.
## The [x,y,z][min,max] values above
## can be used to clip out the region that
## contains the sphere of clouds.
rclouds = 5*kpc
xorigcl = 0*kpc
yorigcl = 0*kpc
zorigcl = 0*kpc



## ~~~~~~~~~~~~~~~~~~~~
## physical properties
## --------------------

namb = 1.38e-3*mp/(mu*amu)
#namb = 1.0

tempamb = 0.68*1.e3*ec*1.e7/kBolz
#tempamb = 1.e7

ncloudsav = 1.
tcut = 3.e4


## ===================
## Do the calculations
## ~~~~~~~~~~~~~~~~~~~

## cellsizes
cellsizex = (cubexmax - cubexmin)/res
cellsizey = (cubeymax - cubeymin)/res
cellsizez = (cubezmax - cubezmin)/res

## Corresponding min/max indices 
## of a shape 3 array of the cube
ixmin = int(res*(xmin - cubexmin)/cubesize)
ixmax = int(res*(xmax - cubexmin)/cubesize)
iymin = int(res*(ymin - cubeymin)/cubesize)
iymax = int(res*(ymax - cubeymin)/cubesize)
izmin = int(res*(zmin - cubezmin)/cubesize)
izmax = int(res*(zmax - cubezmin)/cubesize)

## Corresponding slice position index
if sliceaxnorm == 'z':
    sliceipos = int(res*(slicepos - cubezmin)/cubesize)
elif sliceaxnorm == 'y':
    sliceipos = int(res*(slicepos - cubeymin)/cubesize)
elif sliceaxnorm == 'x':
    sliceipos = int(res*(slicepos - cubexmin)/cubesize)
    

## Read in the cube
cube = np.fromfile(fname,dtype=np.dtype('>f8'),count=-1)

## Reshape into 3D array
cube3 = cube.reshape(res, res, res)

## The cutoff density
ncut = namb*(tempamb/tcut)

## Cut out the cuboid section over which we sum
## This is to reduce computation
cuboid3 = np.split(cube3, [ixmin,ixmax], axis=2)[1]
cuboid3 = np.split(cuboid3, [iymin,iymax], axis=1)[1]
cuboid3 = np.split(cuboid3, [izmin,izmax], axis=0)[1]

## Make copies for arrays holding densities of
## the two phase medium, and just the clouds
## respectively.
cuboid3_2p = cuboid3.copy()
cuboid3_cl = cuboid3.copy()
if rapid:
    np.putmask(cuboid3_2p, cuboid3_2p*ncloudsav < ncut, namb)
    np.putmask(cuboid3_cl, cuboid3_cl*ncloudsav < ncut, 0)


## For cuboid volume 
if volume == 'cuboid' and not rapid:
    
    ## Go through all cells individually
    for k in np.arange(cuboid3_2p.shape[0]):
        for j in np.arange(cuboid3_2p.shape[1]):
            for i in np.arange(cuboid3_2p.shape[2]):
                
                ## Retain reference to ncloudsav for smoothing
                ## nclav will be used to apodize
                nclav = ncloudsav
                
                ## Smoothing if on
                if smooth:
                    h = (i + 0.5)*cellsizex
                    if (h >= xmax - outer_rim_width and h < xmax):
                        tanhfactor = np.tanh(np.tan(np.pi*(h - \
                                     0.5*(2*xmax - outer_rim_width))/ \
                                                   outer_rim_width))
                        
                        ## modify ncloudsav with smoothing factor
                        nclav = (namb + ncloudsav)/2.0 + \
                                tanhfactor*(namb - ncloudsav)/2.0

                ## Apodize cube with cloud density and halo profile                    
                if cuboid3_2p[k][j][i]*nclav < ncut:
                    cuboid3_2p[k][j][i] = namb
                    cuboid3_cl[k][j][i] = 0
                else:                        
                    cuboid3_2p[k][j][i] = cuboid3_2p[k][j][i]*nclav
                    cuboid3_cl[k][j][i] = cuboid3_cl[k][j][i]*nclav


## For spherical volume geometry we will need manipulate manually the outer bound
if volume == 'spherical':

    for k in np.arange(cuboid3_2p.shape[0]):
        for j in np.arange(cuboid3_2p.shape[1]):
            for i in np.arange(cuboid3_2p.shape[2]):
                
                ## radial distance from origin to cell
                rdist = np.sqrt(((i + 0.5)*cellsizex)**2 + \
                                ((j + 0.5)*cellsizey)**2 + \
                                ((k + 0.5)*cellsizez)**2)
                
                ## Set values beyond clouds radius
                if (rdist > rclouds):
                    cuboid3_2p[k][j][i] = namb
                    cuboid3_cl[k][j][i] = 0
                else:
                    ## Apply effects for interior region here.
                    pass
                                
                ## Retain reference to ncloudsav for smoothing
                ## nclav will be used to apodize
                nclav = ncloudsav
                
                ## Smoothing
                if smooth and not rapid:
                    if (rdist >= rclouds - outer_rim_width and rdist < rclouds):
                        tanhfactor = np.tanh(np.tan(np.pi*(rdist - \
                                    0.5*(2*rclouds - outer_rim_width))/ \
                                                     outer_rim_width))
                        nclav = (namb + ncloudsav)/2 + \
                                    tanhfactor*(namb - ncloudsav)/2
                        
                #Apodize cube with cloud density and halo profile                    
                if cuboid3_2p[k][j][i]*nclav < ncut:
                    cuboid3_2p[k][j][i] = namb
                    cuboid3_cl[k][j][i] = 0
                else:                        
                    cuboid3_2p[k][j][i] = cuboid3_2p[k][j][i]*nclav
                    cuboid3_cl[k][j][i] = cuboid3_cl[k][j][i]*nclav





## ~~~~~~~~~~~~~~~
## Calculate mass
## ---------------

if mcalc:

    ## volume of a cell
    cellvol = cellsizex*cellsizey*cellsizez

    ## Total mass in clouds in solar masses.
    ## Use ufunc sum see help(ufunc). 
    mtot_2p = np.sum(cuboid3_2p)*cellvol*mu*amu/(10**9*Msun)
    mtot_cl = np.sum(cuboid3_cl)*cellvol*mu*amu/(10**9*Msun)

    print('Combined mass in both phases = %5.4g M_sun,9' %(mtot_2p) )
    print('Mass in cold phases = %5.4g M_sun,9' %(mtot_cl) )


## Create a slice for visualization
if look:
    if sliceaxnorm == 'z':
        cuboidslice = cuboid3_2p[slicepos][:][:]
    elif sliceaxnorm == 'y':
        cuboidslice = cuboid3_2p[:][slicepos][:]
    elif sliceaxnorm == 'x':
        cuboidslice = cuboid3_2p[:][:][slicepos]

    ## Draw the slice
    pl.imshow(cuboidslice)


#if __name__ == "__main__":
#    plotslice()




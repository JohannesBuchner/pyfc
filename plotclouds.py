import numpy as np
import matplotlib.pylab as pyl
import os.path as osp

import gc

import clouds as cl


## Original cubes can be found on /priv/myriad3/ralph/cubes

#dirpath = '/priv/myriad3/ayw/research/clouds/cubes/'
dirpath = '/Users/ayw/research/clouds/cubes/'
sigma = 5.0
alpha = 5./3.
cubes = [
    #('k40km_1024/k40km_1024.dat', 40, sigma, alpha, [1024,1024,1024]),
    #('k40km_1024/k40a1.7v5.0_1024F64_L00P08.dat', 40, sigma, alpha, [1024,1024,128]),
    #('../k40a1.7v5.0_1024F64_L00P01.dat', 40, sigma, alpha, [1024,1024,1024]),
    #('k20km_1024/k20a1.7v5.0_1024F64_L00P08.dat', 20, sigma, alpha, [1024,1024,128]),
    #('k20km_512/k20a1.7v5.0_512F64_L00P01.dat', 20, sigma, alpha, [512,512,512]),
    ('k01km_32_F64/k01a1.7v5.0_32F64_L00P01.dat', 1, sigma, alpha, [32,32,32]),
    #('k20km_512/v01/k20a1.7v5.0_512F64_L00P08.dat', 20, sigma, alpha, [512,512,64]),
    #('k20km_512/k20km_512.dat', 20, sigma, alpha, [512,512,512]),
    #('k12km_512F64_3D_L00/k12km_512F64_3D_L00.dat', 12, sigma, alpha, [512,512,512]),
    #('k10km_512/k10km_512.dat', 20, sigma, alpha, [512,512,512]),
    #('k10km_512/k10a1.7v5.0_512F64_L00P08.dat', sigma, alpha, 10, [512,512,64]),
    #('../k10a1.7v5.0_512F64_L00P01.dat', 10, sigma, alpha, [512,512,512]),
    #('k05km_512/k05km_512.dat', 20, sigma, alpha, [512,512,512]),
    #('k04km_512F64_L00/k04km_512F64_L00.dat', 20, sigma, alpha, [512,512,512]),
    #('k03km_512F64_L00/k03km_512F64_L00.dat', 20, sigma, alpha, [512,512,512]),
    #('k02km_512_cen/k02km_512_cen.dat', 20, sigma, alpha, [512,512,512]),
    #('k01km_512_c3en/k01km_512_c3en.dat', 20, sigma, alpha, [512,512,512]),
]


#slice_axes = [0, 1, 2]
slice_axes = [2]
#slice_intercepts = np.array(range(res))
#slice_intercepts = range(512)
slice_intercepts = [16,]
numcubes = len(cubes)
num_intercepts = len(slice_intercepts)
num_axes = len(slice_axes)
num_overlays = 1


for icube, (cn, kmin, res, sig, al) in enumerate(cubes):

    ## Get fractal cube object in current dir
    fc = cl.FractalCube(cn, kmin, res, sig, al, dirpath=dirpath) 
    print('cube: ' + str(icube+1)+' / '+str(numcubes)+' '+str(cn))

    for islice, slices in enumerate(fc.cube_nslices_obj(slice_axes,
                                                        slice_intercepts,
                                                        num_overlays)):
        print('|-slice: ' + str(islice+1)+' / '+str(num_intercepts*num_axes)+' '+str(cn))
        fc.create_slices_overlay_image(slices)


    del fc
    gc.collect()

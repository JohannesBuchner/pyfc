"""
________________________________________________________________
Description

Just quickly create two cubes. For testing.
"""
import pyFC
import matplotlib.pyplot as pl

pl.ion()
fcl = pyFC.LogNormalFractalCube()
fcl.gen_cube()
fcg = pyFC.GaussianFractalCube()
fcg.gen_cube()


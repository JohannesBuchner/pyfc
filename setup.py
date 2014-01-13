from distutils.core import setup

# Work around mbcs bug in distutils. 
# http://bugs.python.org/issue10945
import codecs 
try: 
    codecs.lookup('mbcs') 
except LookupError: 
    ascii = codecs.lookup('ascii') 
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs') 
    codecs.register(func) 

setup(
    name = "pyFC",
    packages = ["pyFC"],
    version = "0.1.1",
    description = "Fractal cube generator, analyzer and visualizer",
    author = "Alexander Y. Wagner",
    author_email = "alexander.y.wagner@gmail.com",
    url = "http://www.ccs.tsukuba.ac.jp/Astro/Members/ayw/code/pyFC/index.html",
    keywords = ["clouds", "astrophysics", "simulation", "hydrodynamics", "cfd",
                "initial conditions", "fractal"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    long_description = """\
pyFC: A Fractal cube generator, analyzer and visualizer
-------------------------------------------------------

pyFC is a module to construct "fractal cubes" that are useful for initial conditions in hydrodynamical simulations representing a dense, inhomogeneous component embedded in a more tenuous smooth background. Examples include clouds in earth's athmosphere or the multiphase interstellar medium in galaxies. 

Classes and functions are provided to create and visualize single-point lognormal, two-point fractal fields, statistics often associated with turbulent, dense condensations.

The method of creating lognormal "fractal cubes" was conceived by `Lewis & Austin (2002)`_ for the modeling of clouds in the earth's atmosphere. The scheme of this code is based on that outlined in their paper.

.. _Lewis & Austin (2002): https://ams.confex.com/ams/11AR11CP/techprogram/paper_42772.htm
"""
)

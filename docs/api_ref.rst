.. sectnum::
   :start: 4

API Reference
=============

.. contents:: On this page...
   :local:
   :backlinks: top

Fractal Cube objects
--------------------

The two main classes dealing with fractal cubes in this module are :class:`pyFC.LogNormalFractalCube` and :class:`pyFC.GaussianFractalCube`, which are both derived from :class:`pyFC.FractalCube` (see :file:`clouds.py`). The following are descriptions of the class and its public members.

.. autoclass:: pyFC.LogNormalFractalCube
   :inherited-members:
   :members:
   :undoc-members:
   :show-inheritance:
   
.. autoclass:: pyFC.GaussianFractalCube
   :inherited-members:
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: pyFC.FractalCube._returner

Utility functions
-----------------

The following are some utility function for quick visualization and inspection of fractal cubes. A function beginning with :func:`plot_` creates a figure and plot, whereas a function beginning with :func:`paint_` paints a plot into axes (and colorbar) provided in the argument: 

.. autofunction:: pyFC.paint_midplane_slice
.. autofunction:: pyFC.plot_midplane_slice
.. autofunction:: pyFC.paint_raytrace
.. autofunction:: pyFC.plot_raytrace
.. autofunction:: pyFC.plot_field_stats
.. autofunction:: pyFC.paint_power_spec
.. autofunction:: pyFC.plot_power_spec
.. autofunction:: pyFC.paint_pdf
.. autofunction:: pyFC.plot_pdf

Manipulation functions
----------------------

Various functions exist which support manipulations of fractal cubes. Note that these exist as members of the cube object itself as well, e.g., :func:`fc.mirror`:

.. autofunction:: pyFC.slice
.. autofunction:: pyFC.tri_slice
.. autofunction:: pyFC.translate
.. autofunction:: pyFC.permute
.. autofunction:: pyFC.mirror
.. autofunction:: pyFC.extract_feature
.. autofunction:: pyFC.lthreshold
.. autofunction:: pyFC.pp_raytrace
.. autofunction:: pyFC.pow
.. autofunction:: pyFC.mult
.. autofunction:: pyFC.write_cube



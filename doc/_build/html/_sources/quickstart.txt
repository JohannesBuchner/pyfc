Quickstart
==========

To install the module::

  pip install pyFC

To upgrade the module::

  pip install --upgrade pyFC

You can get the most recent source code from bitbucket::

  git clone 

To load the pyFC module (making sure pyFC in your ``PYTHONPATH``)::

  import pyFC

Create a fractal cube object and generate a lognormal fractal cube with default parameters::

  fc = pyFC.LogNormalFracalCube()
  fc.gen_cube()

The fractal cube data is in the object member ``fc.cube``. The statistical parameters are also contained as members of the object, ``fc.mean``, ``fc.sigma``, ``fc.kmin``, ``fc.beta``, ``fc.n{ijk}``.

Functions exist to manipulate fractal cube objects: ``pyFC.slice``, ``pyFC.tri_slice``, ``pyFC.translate``, ``pyFC.permute``, ``pyFC.mirror``, ``pyFC.extract_feature``, ``pyFC.lthreshold``, ``pyFC.pp_raytrace``, ``pyFC.mult``, ``pyFC.pow`` . (They are associated with the classes ``FCSlicer``, ``FCAffine``, ``FCExtractor``, ``FCRayTracer``, ``FCDataEditor``, ``FCStats``).

To write the fractal cube as data use the routine::

  pyFC.write_cube(<fname>)



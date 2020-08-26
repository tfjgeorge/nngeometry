Parameter space matrix representations
======================================

All parameter space matrix representations inherit from :class:`nngeometry.object.pspace.PMatAbstract`. This abstract class defines all method that can be used with all representations (with some exceptions!). :class:`nngeometry.object.pspace.PMatAbstract` cannot be instantiated, you instead have to choose one of the concrete representations below.

.. autoclass:: nngeometry.object.pspace.PMatAbstract
    :members:

Concrete representations
========================

NNGeometry allows to switch between representations easily. With each representation comes a tradeof between accuracy and memory/computational cost. If testing a new algorithm, we recommend testing on a small network using the most accurate representation that fits in memory (typically :class:`nngeometry.object.pspace.PMatDense`), then switch to a larger scale experiment, and to a lower memory representation.

.. automodule:: nngeometry.object.pspace
    :members:
    :exclude-members: PMatAbstract

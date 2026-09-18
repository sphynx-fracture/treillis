.. currentmodule:: treillis.mechanics


Mechanics
=========

.. toctree::
   :hidden:
   
   mechanics.Element
   mechanics.Fuse
   mechanics.Spring
   mechanics.Beam
   mechanics.shear_bend
   mechanics.isbroken
   mechanics.tensile_fracture
   mechanics.compr_fracture
   mechanics.equivalent_crack
   mechanics.inertia
   mechanics.compliance


*treillis* was originally created for FEM-like mechanical testing on lattices. The ``mechanics`` module is where mechanical testing is performed.

The good practice to import this module is to call it with 

.. code-block:: python
   
   import treillis.mechanics as meca
   
   
Mechanical element
------------------

To perform mechanical tests, element networks are created from lattices, with additionnal data related to the mechanical behavior of the individual elements. Three base element types have been implemented, from simplest to most complex: ``Fuse`` which mimics random-fuse lattices, ``Spring`` with linear elasticity, and ``Beam`` based on Timoshenko's or Euler's beam model.


.. autosummary::
   
   Element
   Fuse
   Spring
   Beam

Utilitaries
-----------

Some functions have been defined to simplify calculations of the mechanical behavior of element networks

.. autosummary::
   
   isbroken
   equivalent_crack
   inertia

Mechanical testing
------------------

The element networks can be tested to recover local and global mechanical properties.

.. autosummary::
   
   shear_bend
   tensile_fracture
   compr_fracture
   compliance

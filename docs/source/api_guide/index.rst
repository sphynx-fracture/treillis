.. module:: treillis

treillis reference
==================

This documentation details all the modules of *treillis*, describing the different classes and useful functions.

.. toctree::
   :hidden:
   
   Lattices <Generation>

.. toctree::
   :hidden:
   :maxdepth: 2
   
   Display

.. toctree::
   :hidden:
   :maxdepth: 2
   
   Mechanics

.. currentmodule:: treillis

*treillis* has a :doc:`main submodule <Generation>`, and several additional submodules. The generation and modification of lattices requires only the main namespace, but pre-coded display and analysis uses other submodules.

Main modules: Generation and modification of lattices
-----------------------------------------------------

Generation of lattices
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   
   makepacking <generation.makepacking>
   frompacking <generation.frompacking>
   Lattice <generation.Lattice>
   Density <generation.Density>
   Density2 <generation.Density2>

.. _modification:

Modification of lattices
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   
   add_elements <modification.add_elements>
   flat_bounds <modification.flat_bounds>
   cut_2Dshape <modification.cut_2Dshape>
   cutbrick <modification.cutbrick>
   cutout <modification.cutout>
   cutsphere <modification.cutsphere>
   mapping <modification.mapping>
   shake <modification.shake>
   
   

Import / Export
^^^^^^^^^^^^^^^

.. autosummary::
   
   saveascii
   savebins
   saveh5
   loadLattice



Special-purpose modules
-----------------------

Displaying lattices
^^^^^^^^^^^^^^^^^^^ 

- :doc:`treillis.display.fast` - Fast visualization and interactive visualization of 3D lattices
- :doc:`treillis.display.clean` - Paper-quality representations of lattices, mainly 2D


Mechanical testing
^^^^^^^^^^^^^^^^^^

Lattice-like objects can be created to perfom :doc:`Mechanics`.
.. currentmodule:: treillis.display.fast


Display
=======

.. toctree::
   :hidden:
   
   treillis.display.fast
   treillis.display.clean

While matplotlib creates beautiful figures that can be saved under various formats which are ideal for papers, the library is extremely slow as soon as the number of plotted points increases over 1000, and its interactivity for 3D plots is less than ideal. Hence, for fast previsualization, and easier plotting of the 3D lattices, we have chosen Datoviz, which is an extremely powerful and fast plotting library in development by Cyrille Rossant. For some basic results on the lattices, matplotlib is still used in the clean plot submodule.

Fast display
------------

The good practice to import this submodule is to call it with 

.. code-block:: python
   
   from treillis.display import fast as fd


``fd`` stands for ``fast_display`` which was previously the standalone module in the 0.1 version of treillis.
The fast plotting functions are the following:

.. autosummary::
   
   colorlat <treillis.display.fast.colorlat>
   displacement <treillis.display.fast.displacement>
   previz <treillis.display.fast.previz>
   scatter_lat <treillis.display.fast.scatter_lat>
   showlat <treillis.display.fast.showlat>
   shownodes <treillis.display.fast.shownodes>

.. currentmodule:: treillis.display.clean


Clean display
-------------

The good practice to import this submodule is to call it with 

.. code-block:: python
   
   from treillis.display import clean as cd

Many functions are common with ``fast display``:

.. autosummary::
   
   colorlat <treillis.display.clean.colorlat>
   displacement <treillis.display.clean.displacement>
   scatter_lat <treillis.display.clean.scatter_lat>
   showlat <treillis.display.clean.showlat>
   shownodes <treillis.display.clean.shownodes>

``showlat`` allows also showing arrows pointing to the boundary nodes ``fixed_nodes``, and permits easier color customization.

Some functions are present in ``clean display`` that are not in ``fast display``:
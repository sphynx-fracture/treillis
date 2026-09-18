Displaying lattice-like objects
===============================

Whether it is to check the design of a lattice, or to observe the results of computations, it is essential to visualize lattices along with various data representations. To simplify visualization, two complementary modules have been implemented:

- :doc:`Fast display <../api_guide/treillis.display.fast>` for fast interactive visualization of the lattices, based on `Datoviz`_ as of 2026. We are very excited for the development of `VisPy 2`_ and `GSP`_ which hopefully will mean fully moving on from the two parallel modules to one single easy to write representation.

.. _Datoviz: https://datoviz.org/
.. _VisPy 2: https://github.com/vispy/vispy2
.. _GSP: https://github.com/vispy/GSP_API/blob/main/whitepaper/gsp-whitepaper.pdf

- :doc:`Clean display <../api_guide/treillis.display.clean>` for article-quality figures, may take very long to plot if you have large lattices, and reduced interactivity due to computation time, but better control over figure sizes and data representation.

Visualization
-------------

Lattices can be observed in a quickly loading (it does scale with the size, with some smart rasterization for saving) setting, with optionally visible axes, boundaries, and :py:class:`~treillis.Domain` box. For ease of analysis, implementation of colored elements and easy scattering over the lattice have been written, as well as a visualization of displacement field. :doc:`fd <../api_guide/treillis.display.fast>` also allows the previsualization of lattices in relief, with the element thickness. Of course, computation of the interactive window gets longer the more elements are in the lattice.
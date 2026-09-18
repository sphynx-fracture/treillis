What is treillis?
=================

*treillis* is a Python library built to help researchers and whoever is interested in microlattices. It provides tools to modelize 2D and 3D microlattices, to compute their elastic properties as well as calculate stresses, strains and elastic constants.

Its writing was initiated in SPEC lab, CEA Iramis, in the woodsy town of Gif-sur-Yvette, France. Its initial use was to model truss-metamaterials in a way that us experimental scientists would be able to use and analyze easily, and enabled us to generate what we call "DBAM" --Delaunay-based metamaterials, which are structurally isotropic.

*treillis* is built using mainly `Numpy`_ and `Scipy`_, as well as `Scikit-sparse`_  for the mechanical calculations. Our goal is to have a package with a limited risk for conflicts between dependencies (which is already difficult enough with `3D visualization`_ ), and an ease-of-use thanks to the `scientific Python ecosystem`_.

.. _Numpy: https://numpy.org/doc/stable/index.html
.. _Scipy: https://scipy.org/
.. _Scikit-sparse: https://scikit-sparse.readthedocs.io/en/latest/
.. _3D visualization: https://datoviz.org/
.. _scientific Python ecosystem: https://scientific-python.org/

Architecture of treillis
------------------------

.. code-block::
   
   treillis/
   ├── docs/
   ├── script_examples/
   └── src/treillis/
      ├── _generation/
      .   ├── __init__.py
      .   ├── prepacking/
      .   .   ├── Packing2D_nodes_0.0.csv
      .   .   ...
      .   ├── 3D-packing-generation/
      .   ├── some_lattices/
      .   .   ├── 28/
      .   .   .   ├── elems.csv
      .   .   .   ├── nodes.csv
      .   .   .   └── parameters.txt
      .   .   ...
      .   ├── meshes.py
      .   ├── packings.py
      .   ├── lattices.py
      .   ├── io.py
      .   ├── redraw.py
      .   └── cutting.py
      ├── display/
      .   ├── __init__.py
      .   ├── plot_utilitaries.py
      .   ├── clean.py
      .   └── fast.py
      └── mechanics/
         ├── __init__.py
         ├── utils.py
         ├── elements.py
         ├── stiffness.py
         └── fracture.py


All the core functions for the lattice generation in the ``_generation`` module are called at the root of *treillis*, meaning it is not necessary to dive into the details of the generation functions if you only need to create the specific lattices that are already pre-defined. All other modules must be called.

Lattices for newbies
--------------------

A :py:class:`~treillis.Lattice` is at the simplest, represented by its :py:attr:`nodes`, which are the connecting dots, linked together by elements or :py:attr:`elems`. To represent each and every lattice, we use a node array of shape (Nnodes, Ndim) to represent the positions in space of the nodes, and an element array of shape (Nelems, 2), which contains the indices of both nodes connected by the given element. Almost everything else depends solely on those two tables and is defined for convenience and speed of use.

Implemented lattice types
^^^^^^^^^^^^^^^^^^^^^^^^^

There are twelve base lattice types, and a composite, whose buildings are implemented in *treillis*.

Periodical lattices
"""""""""""""""""""

.. list-table::
   :align: center
   :width: 80%
   :header-rows: 1
   
   * - Lattice type
     - Name in *treillis*
     - Dimensionality
     - Visualization
   * - Triangular
     - ``'Tri'``
     - 2
     - .. image:: lattices/tri.png
   * - Square
     - ``'Square'``
     - 2
     - .. image:: lattices/square.png
   * - Hexagonal
     - ``'Hexa'``
     - 2
     - .. image:: lattices/hexa.png
   * - Quasi-2D Triangular
     - ``'TriPrism'``
     - 3
     - .. image:: lattices/trip.png
   * - Quasi-2D Hexagonal
     - ``'HexaPrism'``
     - 3
     - .. image:: lattices/hexap.png
   * - Cubic
     - ``'Cubic'``
     - 3
     - .. image:: lattices/cubic.png
   * - Rhombo-hexagonal dodecahedron
     - ``'Octa'``
     - 3
     - .. image:: lattices/octa.png
   * - Rhombic dodecahedron
     - ``'Dodec'``
     - 3
     - .. image:: lattices/dodec.png
   * - Octet-Truss
     - ``'Octet'``
     - 3
     - .. image:: lattices/octet.png


Isotropic lattices
""""""""""""""""""

On top of classic periodical "crystal-like" lattices, *treillis* can generate two types of geometrically isotropic lattices from random close packings: Delaunay-based (with a triangular structure), and Voronoi-based (with an open foam-like structure) lattices, in 2D and 3D.

.. list-table::
   :width: 60%
   :align: center
   :header-rows: 1
   
   * - Lattice type
     - Name in *treillis*
     - Dimensionality
     - Visualization
   * - Delaunay
     - ``'TriDT'``
     - 2
     - .. image:: lattices/tridt.png
   * - Delaunay
     - ``'TriDT3D'``
     - 3
     - .. image:: lattices/tridt3d.png
   * - Voronoi
     - ``'Vor'``
     - 2
     - .. image:: lattices/vor.png
   * - Voronoi
     - ``'Vor3D'``
     - 3
     - .. image:: lattices/vor3d.png


How are lattices generated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Periodic lattices are generated by filling a domain by the elementary cell of their mesh, using translations and rotations. Base available domain are brick-shaped, sphere-shaped, or wedge-splitting shape.

.. figure:: wedge_box.png
   :width: 100%
   :alt: Wedge-splitting.
   :align: center
   
   Typical wedge-splitting geometry, in 3D.


For isotropic lattices, the domain is filled by a random close packing of solid spheres, on the center of whose a triangulation is then applied to recover the final mesh. The 3D random close packing algorithm is based on the `PackingGeneration`_ software developed by Vasili Baranov [#1]_ . The 2D random close packing is inspired by Lozano's `PackGen`_ [#2]_ .

.. _PackingGeneration: https://github.com/VasiliBaranov/packing-generation
.. _PackGen: https://git.tecgraf.puc-rio.br/elozano/packgen

.. rubric:: Footnotes

.. [#] Baranau and Tallarek (**2021**) "Beyond Salsburg–Wood: Glass equation of state for polydisperse hard spheres", *AIP Advances*, 11, DOI: `10.1063/5.0036411`_

.. _10.1063/5.0036411: https://doi.org/10.1063/5.0036411

.. [#] Lozano,  Gattass, (**2016**), "An efficient algorithm to generate random sphere packs in arbitrary domains", *Computers & Mathematics with Applications*, 71 (8), DOI: `10.1016/j.camwa.2016.02.032`_

.. _10.1016/j.camwa.2016.02.032: https://doi.org/10.1016/j.camwa.2016.02.032

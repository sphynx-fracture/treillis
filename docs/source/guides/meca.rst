Mechanical testing
==================

Lattices have initially been created for studying mechanical metamaterials, hence the first application module of treillis is a mechanics module. In first approach, lattices can be considered as FEM meshes, at the element level. Under this framework, we develop a Python-based mechanical testing module.

Understanding the element network
---------------------------------

:py:class:`Elements <treillis.mechanics.Element>` are the objects used to perform mechanical studies, based on lattices. They allow less manipulation over the nodes are they are focused on the mechanics of the elements. Elements can have a :py:attr:`~treillis.mechanics.Element.memory` which is a h5 file that updates with each new performed test, leaving the Python memory relatively empty as only the latest test result is kept in memory, and all the previous ones are saved on your computer hard drive.

Elements have differernt properties depending on the chosen degree of freedom. Three element subclasses have been implemented with increasing number of :py:attr:`~treillis.mechanics.Element.dof`:

- :py:class:`~treillis.mechanics.Fuse` is a random fuse network, where elements have only one degree of freedom, which is their axial stiffness (the linear tension).
- :py:class:`~treillis.mechanics.Spring` is a network of springs, with the possibility to account for flexion in the stiffness, where elements have Ndim degrees of freedom.
- :py:class:`~treillis.mechanics.Beam` is a beam network, where elements have 2Ndim degrees of freedom, accounting for displacement and rotation. There is the possibility to consider the beams in either Euler or Timoshenko framework.

Boundary conditions can be chosen iteratively before applying, either in displacement or in force. The solving is actually always done using the force, zetting the global stiffness matrix to 0 on well-chosen points to recover the displacement after factorization. Factorization is performed using sksparse's :py:func:`Cholesky <sksparse.cholmod.cholesky>` algorithm as much as possible, and falls back to scipy's :py:func:`SuperLU <scipy.sparse.linalg.splu>` if not possible.
So far, all calculation is performed on CPU, but in the future, GPU-acceleration could be used.

Using Elements
--------------

:py:meth:`Applying the chosen boundary conditions to the element network <treillis.mechanics.Element.apply_boundary>` returns the Force on each node, and the mechanical elastic energy. In simple standard tests, these can be used to recover the compliance tensor of the lattice in Voigt notation, and thus recover the effective moduli. 
The full :py:meth:`stress and strain tensors <treillis.mechanics.Beam.stress_strain>` can be approximated over Voronoi elementary volumes defined around each node [#1]_ , and can be used to recover an approximated compliance by performing enough tests to have a fully constrained system (typically, at least 2Ndim). Finally, the fracture of lattices can be studied through iterative tests.

.. rubric:: Footnotes

.. [#] F.Radjai,M.Jean,J.-J.Moreau,S.Roux,Force distributions in dense two-dimensional granular systems. *Physical Review Letters* **77**, 274–277 (1996).
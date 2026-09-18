.. treillis documentation master file, created by
   sphinx-quickstart on Wed Jun  4 17:40:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*treillis* documentation
========================
Site map
--------


*treillis* (/tʁɛji/, French for truss) is a Python library built to help researchers and whoever is interested in microlattices. It provides tools to modelize 2D and 3D microlattices, to compute their elastic properties as well as calculate stresses, strains and elastic constants.

.. grid:: 1
   
    .. grid-item-card::
        :columns: 6
        :text-align: center

        :material-regular:`menu_book;2em;pst-color-primary` **Getting started**
        ^^^

        Find out how to use *treillis* to generate and manipulate lattices through step-by-step user guides.

        +++

        .. button-ref:: guides/index
            :expand:
            :color: secondary
            :click-parent:

            To the user guides

    .. grid-item-card::
        :columns: 6
        :text-align: center

        :material-regular:`developer_mode;2em;pst-color-primary` **API references**
        ^^^

        The complete function references, from the docstring, with detailed descriptions of the parameters and results.

        +++

        .. button-ref:: api_guide/index
            :expand:
            :color: secondary
            :click-parent:

            To the API references

.. card::
   :margin: 0 2 auto auto
   :width: 50%
   :text-align: center
   
   :material-regular:`collections;2em;pst-color-primary` **Example gallery**
   ^^^
   
   A list of examples for various applications of *treillis*, to see more easily how to use it.
   
   +++
   
   .. button-ref:: auto_examples/index
      :expand:
      :color: secondary
      :click-parent:
      
      To the examples


.. toctree::
   :maxdepth: 1
   :hidden:
   
   User Guide <guides/index>
   API references <api_guide/index>
   Examples <auto_examples/index>


About
-----

*treillis* was created during the PhD of Antoine Montiel [#1]_ and has been updated during the PhD of Thibaud Derieux [#2]_. It is maintained by Elina Gilbert of `SPEC lab, CEA`_.

.. _SPEC lab, CEA: https://iramis.cea.fr/en/spec/sphynx/

The library is developed by experimentalists in material sciences, and may show optimization issues and amateurish coding --we gladly welcome suggestions and improvements! It is Python-based even if Python is notably slow, first and foremost because Python is the language that we know best how to code with, but also because it is easy to learn and to use, is widely used in all scientific fields, and has safe implementations of interesting functions written in various other languages, from Fortran to C++ and Julia.

Our aim with this library was primarily to easily generate the objects we are studying, and to correlate our experimental results with easy simulations, but also to provide a Python-based tools for the generation of lattices in a easy-to-handle way that could interest researchers in other domains. We are for example in active discussion with a colleague who specializes in the study of turbulences to possibly add a new submodule. Feel free to write your own, and add it to the library!

.. rubric:: Footnotes

.. [#] Montiel, "Comportement mécanique d'un métamatériau désordonné à base de poutres : vers un microréseau isotrope, rigide, tenace et léger, inspiré de la structure des os", 2022, Université Paris-Saclay, DOI: `10.70675/dc5a8488za51dz4611z8ecbza41f636bd83a`_

.. _10.70675/dc5a8488za51dz4611z8ecbza41f636bd83a: https://doi.org/10.70675/dc5a8488za51dz4611z8ecbza41f636bd83a

.. [#] Derieux, "Conception d'architectures optimales pour métamatériaux isotropes ultra-légers et résistants à la rupture et déformation", 2025, Université Paris-Saclay, DOI: `10.70675/a17f4cfez6074z454ez98d3zc2713dbfdd45`_

.. _10.70675/a17f4cfez6074z454ez98d3zc2713dbfdd45: https://doi.org/10.70675/a17f4cfez6074z454ez98d3zc2713dbfdd45
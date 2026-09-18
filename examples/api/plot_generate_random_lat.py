"""
Generation of a lattice from a Von Mises distribution of points
===============================================================
"""

import numpy as np
from scipy.spatial import Delaunay
import treillis
from treillis.display import clean as cd

# %% 
# Generating points randomly distributed in a plane and linking them together
# with a Delaunay triangulation.

points = np.random.default_rng().vonmises(0, .5, size=[20, 2])
points -= points.mean(axis=0)
DT = Delaunay(points)
e1 = DT.simplices.ravel()
e2 = np.roll(DT.simplices, -1, axis=1).ravel()
links = np.concatenate(
   (e1.reshape(len(e1),1),
    e2.reshape(len(e2),1)),
   axis=1)

# %%
# Assembling the lattice

lat = treillis.Lattice(points, links)
cd.showlat(lat, axis=False, box=False)
print(lat)
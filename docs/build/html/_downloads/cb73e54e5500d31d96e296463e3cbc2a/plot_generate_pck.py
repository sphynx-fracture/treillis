"""
Generating a 2D packing to create a lattice
===========================================
"""

import treillis
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(treillis.style)

pck = treillis.makepacking(2, np.array([30]), 1, .2, False)
# %%
# Observe the packing
plt.figure()
plt.plot(pck[:,0], pck[:,1], '.', markersize=1)
plt.axis('equal')
plt.show()

# %%
# Observe the possible resulting lattice
plt.figure()
plt.triplot(pck[:,0], pck[:,1])
plt.axis('equal')
plt.show()
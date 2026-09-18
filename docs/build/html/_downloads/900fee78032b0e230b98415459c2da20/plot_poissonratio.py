"""
Change lattice elements with Poisson distribution
=================================================
"""
import treillis
from treillis.display import fast as fd
from treillis.display import clean as cd

lat = treillis.loadLattice('lattice_130')
mapped, pos = treillis.mapping(lat, 'poisson', 5, return_pos=True)

# %% 
# View the mapping

cd.colorlat(lat, mapped, label='aspect ratio variations');

# %%
# Changing the aspect ratio based on the mapping

mapped -= mapped.mean()
lat.aspect_ratio *= 1 + mapped

# %%
# View the new lattice

fd.previz(lat)
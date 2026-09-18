"""
Verification of the mechanical isotropy of a lattice
====================================================

This script is a nice twin to the :ref:`structural isotropy test <plot_isotropy>` structural isotropy test, as we perform
some simple mechanical tests over a lattice to evaluate its mechanical
isotropy in the linear domain.
"""

import os
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import colorizer, collections 

import ipyparallel as ipp

import treillis
import treillis.mechanics as meca
from treillis.display import clean as cd

plt.style.use(treillis.style)


# %% 
# Lattice creation
# ----------------

lat = treillis.Lattice.initialize([100,100], 1, 'TriDT', des=.2)

lat = treillis.flat_bounds(lat)
lat.clean_lattice()

cd.showlat(lat, boundary=True)
plt.show()

# %%
# Mechanical testing
# ------------------
# 
# Creation of the sample
# """"""""""""""""""""""

sample = meca.Beam(lat, 1, 1, .4, location='off')

# %%
# Writing the test to be performed
# """"""""""""""""""""""""""""""""
# To test isotropy, the sample is loaded in several directions with unit
#  amplitude. To accelerate testing, we can run them in parallel.
# The displacement is blocked in the orthogonal direaction as "displacement-
# imposed" tests are actually performed by setting the force. By blocking
# orthogonal displacement, we recover the Poisson ratio from the reaction of
# the sample.

val = sample.L
epsilon = [np.array([1.,0])*val, np.array([0,1.])*val, np.array([.5,.5])*val] 
args = [[eps, deepcopy(sample)] for eps in epsilon];

def get_elasticity(args):
    import numpy as np
    import treillis
    import treillis.mechanics as meca
    
    eps, beam = args
    
    fixed = beam.fixed_nodes
    pos = beam.nodes[fixed]
    displ = np.ones((fixed.shape[0], beam.dof)) * np.nan
    right = np.flatnonzero(pos[:,0].max() - pos[:,0] < beam.L/2)
    left = np.flatnonzero(pos[:,0] - pos[:,0].min() < beam.L/2)
    top = np.flatnonzero(pos[:,1].max() - pos[:,1] < beam.L/2)
    bottom = np.flatnonzero(pos[:,1] - pos[:,1].min()< beam.L/2)
    
    if eps[1]==0:
        displ[right,0] = eps[0] + pos[right, 0].max() - pos[right, 0]
        displ[left, 0] = 0
        displ[top,1] = 0
        displ[bottom,1] = 0
    elif eps[0]==0:
        displ[top,1] = eps[1] + pos[top, 1].max() - pos[top, 1]
        displ[bottom,1] = 0
        displ[right, 0] = 0
        displ[left,0]=0
    else:
        displ[right,1] = eps[0]
        displ[top,0] = eps[1]
        displ[left, 1] = 0
        displ[bottom, 0] = 0
    
    
    beam.boundary_condition(beam.fixed_nodes, displ, reset=True)
    E, F = beam.apply_boundary()
    res = beam.stress_strain()
    
    return E, F, beam.displacement, res
    
Eelastic, Force, new_nodes, Tensors = [], [], [], []
with ipp.Cluster(n=int(os.cpu_count()*2/3)) as rc:
    dview = rc[:]
    
    res = dview.map_async(get_elasticity, args);
    res.wait_interactive()
    for r in res:
        Eelastic.append(r[0])
        Force.append(r[1])
        new_nodes.append(sample.nodes + r[2][:,:2])
        Tensors.append(r[3])
        
del res, dview

# %%
# Experience display
# """"""""""""""""""
# Displacement
# ^^^^^^^^^^^^

fig, ax = plt.subplots(ncols=3,
                       figsize = (3.2*cd.xplot, cd.xplot),
                       gridspec_kw={'wspace': cd.xplot/15,
                                    'hspace': cd.xplot/15})

for i in range(3):    
    norm = mc.Normalize(vmin=np.min(np.sqrt(np.nansum((new_nodes[i]-lat.nodes)**2, axis=1))),
                        vmax=np.max(np.sqrt(np.nansum((new_nodes[i]-lat.nodes)**2, axis=1))))
    ax[i].quiver(lat.nodes[:,0], lat.nodes[:,1],
                 new_nodes[i][:,0]-lat.nodes[:,0], 
                 new_nodes[i][:,1]-lat.nodes[:,1],
                 scale=1, scale_units='xy',
                 color=cd.maparc(norm(np.sqrt(np.sum((new_nodes[i]-lat.nodes)**2, axis=1)))))
    
    ax[i].set_xlim([-lat.sizedom[0]/2-2*val,lat.sizedom[0]/2+2*val]);
    ax[i].set_ylim([-lat.sizedom[1]/2-2*val,lat.sizedom[1]/2+2*val]);
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    
ax[0].set_title(r'$\varepsilon=[1,0,0]$', **cd.titlesfont)
ax[1].set_title(r'$\varepsilon=[0,1,0]$', **cd.titlesfont)
ax[2].set_title(r'$\varepsilon=[0,0,1]$', **cd.titlesfont)
                                  
fig.suptitle('Test displacements', y=1.05, fontsize=30, **cd.titlesfont)
plt.show()


# %%
# Force
# ^^^^^

fig, ax = plt.subplots(ncols=3,
                       figsize = (3.2*cd.xplot, cd.xplot),
                       gridspec_kw={'wspace': cd.xplot/15})

for i in range(3):    
    F = np.sqrt((Force[i][::3] + Force[i][1::3])**2)
    norm = mc.Normalize(vmin=np.nanmin(F),
                        vmax=np.nanmax(F))
    ax[i].quiver(lat.nodes[:,0], lat.nodes[:,1],
                 Force[i][::3], Force[i][1::3],
                 color=plt.colormaps['jet'](norm(F)))
    
    ax[i].set_xlim([-lat.sizedom[0],lat.sizedom[0]]);
    ax[i].set_ylim([-lat.sizedom[1],lat.sizedom[1]]);
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    
ax[0].set_title(r'$\varepsilon=[1,0,0]$', **cd.titlesfont)
ax[1].set_title(r'$\varepsilon=[0,1,0]$', **cd.titlesfont)
ax[2].set_title(r'$\varepsilon=[0,0,1]$', **cd.titlesfont)

fig.suptitle('Test Forces', y=1.05, fontsize=30, **cd.titlesfont)
plt.show()

# %%
# Calculation of the compliance from the force
# """"""""""""""""""""""""""""""""""""""""""""

pos = sample.nodes[sample.fixed_nodes]
right = np.flatnonzero(pos[:,0].max() - pos[:,0] < 2*sample.L)
top = np.flatnonzero(pos[:,1].max() - pos[:,1] < 2*sample.L)

force = [f*sample.sizedom[0]/val*sample.L/sample.aspect_ratio.mean()
         for f in Force]

Cxx = np.nansum(force[0][::3][sample.fixed_nodes[right]])*2
Cyx = force[0][1::3][sample.fixed_nodes[top]].sum()
Cgx = np.nansum(force[0][1::3][sample.fixed_nodes[right]])*2

Cyy = np.nansum(force[1][1::3][sample.fixed_nodes[top]])*2
Cxy = force[1][::3][sample.fixed_nodes[right]].sum()
Cgy = np.nansum(force[1][::3][sample.fixed_nodes[top]])*2

Cgg1 = force[2][1::3][sample.fixed_nodes[right]].sum()
Cgg2 = force[2][::3][sample.fixed_nodes[top]].sum()
Cxg = force[2][::3][sample.fixed_nodes[right]].sum()
Cyg = force[2][1::3][sample.fixed_nodes[top]].sum()


C = np.array([[Cxx,    Cxy,    Cxg],
              [Cyx,    Cyy,    Cyg],
              [Cgx, Cgy, (Cgg1+Cgg2)]])

# %%%
# If the sample is orthotropic:
    
Gxy = (Cgg1+Cgg2)/2
nuxy = Cyx/Cxx
nuyx = Cxy/Cyy
Ex = (Cxx + Cxy/nuyx)*(1-nuxy*nuyx)/2
Ey = (Cyy + Cyx/nuxy)*(1-nuxy*nuyx)/2

# %%%
# If the sample is isotropic
nu = (Cyx/Cxx + Cxy/Cyy)
E = (Cxx + Cyy) * (1-nu**2)/2

# %%%
# Comparison of orthotropic and isotropic results

if abs(nuxy-nuyx) < nuxy/10:
    print("The sample is isotropic, with a Young's modulus of",np.round(E,2), 
          "Pa and a Poisson's ratio of", np.round(nu,2))
else:
    print("The sample is not isotropic, and the values from an orthotropic "
          +"hypothesis are Ex =", np.round(Ex,2), "Ey =", np.round(Ey,2),
          "nu_xy =", np.round(nuxy,2), "and nu_yx =", np.round(nuyx,2))
    
# %%
# Visualization of approximated stress and strain
# """""""""""""""""""""""""""""""""""""""""""""""

fig, ax = plt.subplots(ncols=2, nrows=3, 
                       figsize = (3*cd.xplot, 4*cd.xplot),
                       gridspec_kw={'hspace': cd.xplot/15, 
                                    'wspace': cd.xplot/13})

for i in range(6):    
    # Strain
    if not i%2 :
        t = np.linalg.norm(Tensors[i//2]['local strain'], axis=(1,2))
        norm = plt.Normalize(t.min(), t.max())
        
        col = collections.PolyCollection(
            [Tensors[i//2]['deformed vertices'][j] 
             for j in range(len(Tensors[i//2]['vertices']))],
            color=cd.mapwis(norm(t)),
            rasterized=True)
        
        ax[i//2,0].add_collection(col)

        cr = colorizer.Colorizer(norm=norm, cmap=cd.mapwis)
        fig.colorbar(colorizer.ColorizingArtist(cr), ax=ax[i//2,0])
    
    # Stress
    else:
        t = np.linalg.norm(Tensors[i//2]['local stress'], axis=(1,2))
        norm = plt.Normalize(t.min(), t.max())
        
        col = collections.PolyCollection(
            [Tensors[i//2]['deformed vertices'][j] 
             for j in range(len(Tensors[i//2]['vertices']))],
            color=cd.mapope(norm(t)),
            rasterized=True)
        
        ax[i//2,1].add_collection(col)
        
        cr = colorizer.Colorizer(norm=norm, cmap=cd.mapope)
        fig.colorbar(colorizer.ColorizingArtist(cr), ax=ax[i//2,1])
    
    ax[i//2, i%2].set_xlim([-lat.sizedom[0]/2-2*val,lat.sizedom[0]/2+2*val]);
    ax[i//2, i%2].set_ylim([-lat.sizedom[1]/2-2*val,lat.sizedom[1]/2+2*val]);
    ax[i//2, i%2].set_xticks([])
    ax[i//2, i%2].set_yticks([])
                                  
ax[2,0].set_xlabel(r'||$\varepsilon$||', **cd.titlesfont);
ax[2,1].set_xlabel(r'||$\sigma$||', **cd.titlesfont);
ax[0,0].set_ylabel(r'$\varepsilon=[1,0,0]$', **cd.titlesfont)
ax[1,0].set_ylabel(r'$\varepsilon=[0,1,0]$', **cd.titlesfont)
ax[2,0].set_ylabel(r'$\varepsilon=[0,0,1]$', **cd.titlesfont)

fig.suptitle('Test results', y=.93, fontsize=30, **cd.titlesfont)
plt.show()

"""Follow along the fracture of a lattice in a tensile test."""
from time import time
import numpy as np

import matplotlib.pyplot as plt

import treillis
import treillis.mechanics as meca

#%% Generate a lattice in the pure shear configuration (thin strip)
l0 = 1.
size = np.array([30*l0, 10*l0, 5*l0])
size += l0*np.sqrt(3)
lat = treillis.Lattice.initialize(size, l0, 'TriDT3D')
lat.clean_lattice()
# lat = treillis.flat_bounds(lat)
# lat.clean_lattice()

# fd.showlat(lat)

#%% Generate the elements network from the lattice for the actual testing
Fc = np.random.default_rng().weibull(4, size=lat.elems.shape[0])
t0 = time()
beams = meca.Beam(lat, Fc, 1, .4, 
                  location=r'C:\Users\EG283299\Documents\04-Codes\Python'
                  + r'\Lattices\Antoine')
tf = time()
print('Beams generated in', np.round(tf-t0, 2),  's.')
#%%% network properties
# Introduce an initial crack in the sample
beams.introduce_crack(-beams.sizedom[0]/2+(1/3) * lat.sizedom[0], 0)
# fd.showlat(beams)
ic = beams.crack.shape[0]

#%%% Choose the boundary conditions
traction = lat.sizedom[1]/2
upper_pos = np.max(beams.nodes[beams.fixed_nodes,1])
up_nodes = np.flatnonzero(beams.nodes[beams.fixed_nodes,1]
                          >= upper_pos - beams.L/3)
up_nodes = beams.fixed_nodes[up_nodes]

lower_pos = np.min(beams.nodes[beams.fixed_nodes,1])
lo_nodes = np.flatnonzero(beams.nodes[beams.fixed_nodes,1]
                          <= lower_pos + beams.L/3)
lo_nodes = beams.fixed_nodes[lo_nodes]

beams.boundary_condition(up_nodes, np.array([np.nan, traction, np.nan, 
                                             np.nan, np.nan, np.nan]))
beams.boundary_condition(lo_nodes, np.array([np.nan, 0, np.nan, 
                                             np.nan, np.nan, np.nan]))

#%% Solving with the initial conditions
E, F = beams.apply_boundary()

#%% Fracture
#%%% Prepare for cycling

def get_K_indx(nodes, dof):
    index = np.repeat(nodes, dof)
    for n in nodes:
        if len(index[index==n])==dof:
            index[index==n] = np.array([dof*n + i
                                        for i in range(dof)])
        else:
            index[index==n] = np.concatenate((
                index[index==n][:-dof], 
                np.array([dof*n + i for i in range(dof)])))
    return index

bound_up = get_K_indx(up_nodes, beams.dof)
bound_lo = get_K_indx(lo_nodes, beams.dof)

lc = (1/6) * beams.sizedom[0]

main_crack = (beams.nodes[beams.crack[:,0].astype(int)]
              + beams.nodes[beams.crack[:,1].astype(int)]) / 2

#%%% Loop for fracture propagation

broken = False
N_broken = 0
Energy = []
Force = []
K = [[],[]]
U = [[],[]]
F = []
length = []

print('Starting fracture propagation')
t0 = time()

# Normally, the length condition is sufficient but better safe than infinite loop!
while not broken and lc < 2/3 * size[0]:
    M, V = meca.shear_bend(beams)
    N = -beams.force.ravel()
    indx_broken, factor = meca.tensile_fracture(beams, N, M=M, V=V)
    broken = meca.isbroken(beams, indx_broken)
    
    # Get properties just before crack propagation
    Unew = np.ravel(beams.displacement / factor)
    force_nodes = beams.K_global.dot(Unew)
    F_up = np.sum(force_nodes[bound_up[1::beams.dof]])
    K_before = F_up / (2 * traction / factor)
    Upot_before = 0.5 * np.transpose(Unew).dot(beams.K_global.dot(Unew))

    K[0].append(K_before)
    U[0].append(Upot_before)

    # Propagate the crack
    beams.break_elem(indx_broken)
    N_broken += 1
    
    # Recalculate the new initial state stiffness with the propagated crack
    beams.reassemble()
    
    # Apply new traction normalized to bo to traction
    beams.boundary_condition(up_nodes, np.array([np.nan, traction/factor, np.nan, 
                                                 np.nan, np.nan, np.nan]))
    beams.boundary_condition(lo_nodes, np.array([np.nan, 0, np.nan, 
                                                 np.nan, np.nan, np.nan]))
    
    e, f = beams.apply_boundary()
    Energy.append(e)
    Force.append(f[bound_up[1::beams.dof]])
    
    # Get properties just after traction
    Unew = np.ravel(beams.displacement / factor)
    F_up = np.sum(f)
    K_after = F_up / (2 * traction / factor)
    Upot_after = e
    F.append(factor)
    
    beams.displacement *= factor
    beams.force *= factor
    
    K[1].append(K_after)
    U[1].append(Upot_after)
    
    
    # Not the best way to get the crack length, until we have a better 
    # equivalent crack length function.
    D_crack = np.sqrt(np.sum(
        (main_crack
         - (beams.nodes[beams.crack[-1,0].astype(int)] 
            + beams.nodes[beams.crack[-1,1].astype(int)]) / 2)
        **2, axis=1))
    if len(D_crack[D_crack <= beams.L]) >= 2:
        main_crack = np.concatenate((
            main_crack,
            ((beams.nodes[beams.crack[-1,0].astype(int)] 
             + beams.nodes[beams.crack[-1,1].astype(int)])/2).reshape(
                 1, beams.dim)),
            axis = 0)
    
    lc = max(main_crack[:,0]) + beams.sizedom[0]/2
    length.append(lc)
    
    # Plotting every ten steps and reducing the memory cost
    if N_broken%10 == 0:
        plt.figure()
        crackpos = (beams.nodes[beams.crack[:,0].astype(int)]
                    + beams.nodes[beams.crack[:,1].astype(int)]) / 2
        col = np.concatenate((
            np.zeros(ic),
            np.arange(ic,crackpos.shape[0])-ic+1))
        plt.scatter(crackpos[:,0], crackpos[:,1],
                    s = (crackpos[:,2]+beams.sizedom[2]/2)/beams.sizedom[2]*5,
                    c = col, rasterized=True)
        
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((-beams.sizedom[0]/2, -beams.sizedom[1]/2),
                                   beams.sizedom[0], beams.sizedom[1],
                                   fill=False, color='lightgrey'))
        
        plt.xlim([-beams.sizedom[0]/2-beams.L, beams.sizedom[0]/2+beams.L])
        plt.ylim([-beams.sizedom[1]/2-beams.L, beams.sizedom[1]/2+beams.L])
        plt.axis('equal')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top','bottom','left','right']].set_visible(False)
        
        plt.show()
        
        # Keep only every 10 steps for a lighter memory
        to_erase = np.arange(beams.Ntest-20, beams.Ntest-2)
        beams.erase_tests(to_erase)
    
tf = time()

print('Lattice broke after ', np.round(tf-t0,2), 's.')

Energy = np.array(Energy)
Force = np.array(Force)
K = np.array(K)
U = np.array(U)
F = np.array(F)
length = np.array(length)

np.savetxt(r'C:\Users\EG283299\Documents\04-Codes\Python\Lattices\Antoine\energy.csv',
           Energy)
np.savetxt(r'C:\Users\EG283299\Documents\04-Codes\Python\Lattices\Antoine\force.csv',
           Force)
np.savetxt(r'C:\Users\EG283299\Documents\04-Codes\Python\Lattices\Antoine\upot.csv',
           U)
# %% Plotting

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6,8), sharex=True)
ax[0].plot(length, K[:,1] - K[:,0])
ax[1].plot(length, U[:,1] - U[:,0])
ax[1].set_xlabel('Equivalent crack length')
ax[0].set_ylabel('Stiffness difference')
ax[1].set_ylabel('Released energy')
plt.show()

crackprogr = np.concatenate((np.zeros(beams.crack.shape[0] - N_broken),
                             np.arange(N_broken, beams.crack.shape[0])))

# fd.scatter_lat(beams, 
#             (beams.nodes[beams.crack[:,0].astype(int)]
#              + beams.nodes[beams.crack[:,1].astype(int)]) / 2,
#             values=crackprogr)
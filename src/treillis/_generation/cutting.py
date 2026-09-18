"""
Cutting
=======

The objective of this module is to facilitate slicing into lattices.
"""

#%% Package importation

import os
from copy import deepcopy
import numpy as np
from tqdm import trange
import ipyparallel as ipp

from scipy.spatial import Delaunay

from .lattices import Lattice


__all__ = ['cutbrick', 'cutsphere', 'cutout', 'cut_2Dshape', 'isolate']

#%% Small functions to reuse

def keep_elem(elements: np.ndarray, I_kept: np.ndarray):
    """Keep only the elements linked to the nodes in I_kept, and renumber them to corrsepond to the indices of I_kept."""
    elems = deepcopy(elements)
    keep = np.array([np.prod(np.isin(e, I_kept)) for e in elems[:,:2]],
                    dtype=bool)        
    elems = elems[keep]

    # Renumbering the elements
    for i in range(len(I_kept)):
        if np.isin(I_kept[i], elems[:,:2]).any():
            elems[elems[:,0]==I_kept[i],0] = i
            elems[elems[:,1]==I_kept[i],1] = i
    return elems

def keep_elem_prop(elements: np.ndarray, param: np.ndarray, I_kept: np.ndarray):
    """Keep only the element property for the elements linked to the nodes in I_kept."""
    param = deepcopy(param)
    keep = np.array([np.prod(np.isin(e, I_kept)) for e in elements[:,:2]],
                    dtype=bool)        
    param = param[keep]
    return param

def isin(point, shape, sx, sy):
    """Consider a node in the plot if it is surrounded by at least 2 white points."""
    x = np.linspace(-sx/2,sx/2,shape.shape[1])
    y = np.linspace(-sy/2,sy/2,shape.shape[0])
    
    Ix = np.argmin(abs(point[0]-x))
    Iy = np.argmin(abs(point[1]-y))

    x0 = max(Ix-1, 0)
    xf = min(Ix+1, shape.shape[1])

    y0 = max(Iy-1, 0)
    yf = min(Iy+1, shape.shape[0])

    return np.sum(shape[y0:yf, x0:xf])>1

#%% Main functions
#%%% Basic shapes

def cutbrick(lattice: Lattice, xlims: list|None = None,
          ylims: list|None = None, zlims: list|None = None):
    """Create a sliced lattice from within limits (=[[x_min,x_max], [y_min,y_max], [z_min,z_max]])."""
    nodes = deepcopy(lattice.nodes)
    elems = deepcopy(lattice.elems)
    I = np.arange(len(nodes))
    
    if xlims is None: xlims = [-np.inf, np.inf]
    if ylims is None: ylims = [-np.inf, np.inf]

    if xlims[0]<-lattice.sizedom[0]/2 and xlims[1]>lattice.sizedom[0]/2\
        and ylims[0]<-lattice.sizedom[1]/2 and ylims[1]>lattice.sizedom[1]/2:
            if zlims is None: return lattice
            elif zlims[0]<-lattice.sizedom[2]/2\
                and zlims[1]>lattice.sizedom[2]/2: return lattice 

    # For safety: we recalculate the limits to be within the lattice
    xlims = [max(xlims[0], -lattice.sizedom[0]/2),
             min(xlims[1], lattice.sizedom[0]/2)]
    ylims = [max(ylims[0], -lattice.sizedom[1]/2),
             min(ylims[1], lattice.sizedom[1]/2)]


    Ix = I[(nodes[:,0]>=xlims[0]) & (nodes[:,0]<=xlims[1])]
    Iy = I[(nodes[:,1]>=ylims[0]) & (nodes[:,1]<=ylims[1])]

    Ii = np.intersect1d(Ix, Iy)

    if zlims is not None and lattice.dim==3:
        zlims = [max(zlims[0], -lattice.sizedom[2]/2),
                 min(zlims[1], lattice.sizedom[2]/2)]
        Iz = I[(nodes[:,2]>=zlims[0]) & (nodes[:,2]<=zlims[1])]
        If = np.intersect1d(Ii, Iz)
    else: If = Ii

    # Redefining the parameters corresponding to the elems
    ratio = keep_elem_prop(lattice.elems, lattice.aspect_ratio, If)
    if isinstance(lattice.beam_section, np.ndarray):
        section = keep_elem_prop(lattice.elems, lattice.beam_section, If)
    else: section = lattice.beam_section
            
    elems = keep_elem(elems, If)
    return Lattice(lattice.nodes[If,:], elems, length=lattice.L,
                   inimesh=lattice.type, beam_section=section,
                   aspect_ratio=ratio)
    

def cutsphere(lattice: Lattice, pos: np.ndarray = None, rlim: float = None):
    """Cut the lattice in a spherical shape at the given position and radius."""
    if pos is not None:
        pos = np.array(pos).reshape((1, lattice.dim))
    else: pos = np.zeros_like(lattice.nodes)
    nodes = deepcopy(lattice.nodes)
    nodes -= pos
    elems = deepcopy(lattice.elems)
    I = np.arange(len(nodes))
    
    if rlim is not None:
        lmax = np.max(np.sqrt(np.sum(nodes**2, axis=1)))
        if rlim<=lmax:
            cond = np.sqrt(np.sum(nodes**2, axis=1))<=rlim
            If = I[cond]
            
            # Selecting the ROI
            nodes = nodes[If,:]

            # Redefining the parameters corresponding to the elems
            ratio = keep_elem_prop(elems, lattice.aspect_ratio, If)
            if isinstance(lattice.beam_section, np.ndarray):
                section = keep_elem_prop(elems, lattice.beam_section, If)
            else: section = lattice.beam_section

            elems = keep_elem(elems, If)
            return Lattice(nodes, elems, length=lattice.L,
                           inimesh=lattice.type, beam_section=section,
                           aspect_ratio=ratio, size = np.array([rlim]))
    
    print("There is no change in the lattice.")
    return lattice


def cutout(lattice: Lattice, position: np.ndarray, size: float|np.ndarray):
    """
    Cut out a brick or a spherical shape in the lattice.

    Parameters
    ----------
    lattice : Lattice
        Lattice to cut out.
    position : np.ndarray
        Position of the cut. Center of the sphere, or lower back left side of the brick.
    size : float|np.ndarray
        If only one value, radius of the sphere, if more values, respectively lengths along x, y, and z of the brick cutout.

    Returns
    -------
    Lattice
        Lattice with shape cut out.
        
    Examples
    --------
    Cut out the center of a spherical lattice.
    
    >>> import treillis
    >>> lat = treillis.Lattice.initialize(10, 1, 'TriDT3D')
    >>> cutlat = treillis.cutout(lat, [0,0,0], 8)
    """
    nodes = deepcopy(lattice.nodes)
    elems = deepcopy(lattice.elems)
    elmpos = np.concatenate((
        nodes[elems[:,0].astype(int)]\
        .reshape(elems.shape[0],lattice.dim,1),
        nodes[elems[:,1].astype(int)]\
        .reshape(elems.shape[0],lattice.dim,1)), axis=2)
    shape = 'brick'
        
    try: len(position) == lattice.dim
    except: raise ValueError("Initial position cannot be a float.")
    position = np.array(position)
    if len(position) != lattice.dim:
        raise ValueError("Lattice is ", lattice.dim,
                             "D, but initial position is ", len(position),
                             "D.")

    if not isinstance(size, (list, np.ndarray)): shape = 'sphere'
    elif len(size) == 1: shape = 'sphere'
    
    I = np.arange(elmpos.shape[0])
    
    if shape=='sphere':
        cond = (np.sqrt(np.sum((elmpos[:,:,0]-position)**2, axis=1)) <= size)\
            + (np.sqrt(np.sum((elmpos[:,:,1]-position)**2, axis=1)) <= size)
        cond = cond.astype(bool)
    else:
        condx = (elmpos[:,0,:] >= position[0])\
            * (elmpos[:,0,:]-position[0] < size[0]) 
        condy = (elmpos[:,1,:] >= position[1])\
            * (elmpos[:,1,:]-position[1] < size[1])
        
        cond = condx * condy
        
        if len(size) == 3:
            if lattice.dim == 3:
                condz = (elmpos[:,2,:] >= position[2])\
                    * (elmpos[:,2,:]-position[2] < size[2])
                cond = cond * condz
            else:
                print("Warning: size is in 3D, but lattice is only 2D: third dimension will be ignored.")
                
        cond = np.sum(cond, axis=1, dtype=bool)
        
        if len(size) < len(position):
            S = np.zeros(position.shape)
            S[:len(size)] = size
            size = S
        else: size = np.array(size)
        
        trans = (elmpos[:,:,0] < np.repeat(position.reshape(1, lattice.dim),
                                          elmpos.shape[0], axis=0))\
            * (elmpos[:,:,1] >= np.repeat(position.reshape(1, lattice.dim),
                                              elmpos.shape[0], axis=0)\
               + np.repeat(size.reshape(1, lattice.dim),
                           elmpos.shape[0], axis=0))
        trans = np.sum(trans, axis=1, dtype=bool)
        
        cond = cond + trans
        
    keep = np.delete(I, cond)
    elmpos = elmpos[keep]
    
    ratio = lattice.aspect_ratio[keep]
    if isinstance(lattice.beam_section, np.ndarray):
        section = lattice.beam_section[keep]
    else: section = lattice.beam_section
    
    nodes = np.concatenate((elmpos[:,:,0], elmpos[:,:,1]), axis=0)
    nodes = np.unique(nodes, axis=0)
    
    Inodes = np.arange(nodes.shape[0])
    
    elems = np.concatenate((
        np.array([[Inodes[np.prod(nodes == e, axis=1, dtype=bool)]]
                   for e in elmpos[:,:,0]]),
        np.array([[Inodes[np.prod(nodes == e, axis=1, dtype=bool)]]
                   for e in elmpos[:,:,1]])), axis=1)
    elems = elems.reshape(elems.shape[0], 2)
    
    return Lattice(nodes, elems, length=lattice.L,
                   inimesh=lattice.type, beam_section=section,
                   aspect_ratio=ratio, size = lattice.sizedom)
        
        
        

def isolate(lattice: Lattice):
    """Clusterize a cut lattice in a list of sub-lattices."""
    node_list = np.arange(lattice.nodes.shape[0])
    Lattices = []
    
    contacts = lattice.contacts_list
    
    print("Getting the separate clusters")
    while len(node_list) > 0:
        c0 = node_list[0]
        Lattices.append(np.array([c0]))
        l = contacts[c0]
        l = l[np.isin(l, Lattices[-1], invert=True)]
        while len(l>0):
            Lattices[-1] = np.concatenate((Lattices[-1], l))
            Lattices[-1] = np.unique(Lattices[-1])
            
            l = np.concatenate([contacts[i] for i in l])
            l = np.unique(l)
            l = l[np.isin(l, Lattices[-1], invert=True)]
                                          
        print('Found', len(Lattices), 'clusters')
        node_list = node_list[np.isin(node_list, Lattices[-1], 
                                      invert=True)]
        
    if len(Lattices)==1:
        print("No sub-lattice found")
        return lattice
        
    print("Creating the sub-lattices")
    lat_list = []
    for i in trange(len(Lattices)):
        idx = Lattices[i]
        mask_elm = np.prod(np.isin(
            lattice.elems[:,:2], idx), axis=1, dtype=bool)
        
        elm = keep_elem(lattice.elems, idx)
        
        if type(lattice.beam_section)!=str: 
            beam = lattice.beam_section[mask_elm]
        else: beam = lattice.beam_section
        
        lat = Lattice(lattice.nodes[idx],
                      elm,
                      size=lattice.sizedom, length=lattice.L,
                      inimesh=lattice.type,
                      aspect_ratio=lattice.aspect_ratio[mask_elm],
                      beam_section=beam)
        
        lat_list.append(lat)
        
    return lat_list
        
    
#%%% Fun shapes
def cut_2Dshape(lattice: Lattice, shape: np.ndarray, 
                sx: float = None, sy: float = None):
    """
    Cutout a 2D shape described by a black-and-white image in a lattice.
    
    By convention, the shape will be cutout in the XY plane.

    Parameters
    ----------
    lattice : Lattice
        The lattice you want to cut
    shape : 2D np.ndarray
        A black and white image encoded in 0s and 1s. By convention, the nodes
        that are kept are in the
    sx : float, optional
        the size along the x-axis of the image. If not given, will be the
        width of the lattice.
    sy: float, optional
        the size along the y-axis of the image. If not given, will be the height 
        of the lattice.

    Returns
    -------
    Lattice
    
    Examples
    --------
    Cutting a star-shape in a lattice.
    
    >>> import numpy as np
    >>> import treillis
    >>> # create a lattice
    >>> lat = treillis.Lattice.initialize([50, 50], 2, 'Vor', des=.1)
    >>> # create the cutting template
    >>> star = np.array([
    >>>     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
    >>>     [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
    >>>     [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    >>>     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    >>>     [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    >>>     [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    >>>     [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
    >>>     [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    >>>     [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    >>>     [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    >>>     [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    >>>     [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    >>>     [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    >>>     [0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0],
    >>>     [0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0],
    >>>     [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0]])
    >>> starlat = treillis.cut_2Dshape(lat, star)
    
    Because of the way the node position is implemented, the star will be
    upside-down in the lattice.
    """
    nodes = deepcopy(lattice.nodes)
    elems = deepcopy(lattice.elems)
    if sx==0 or sy==0: raise ValueError("Cannot cut a 1D line in a lattice.")
    if sx is None: sx = np.max(lattice.nodes[:,0])-np.min(lattice.nodes[:,0])
    if sy is None: sy = np.max(lattice.nodes[:,1])-np.min(lattice.nodes[:,1])

    I = np.arange(len(lattice.nodes))

    I_ok = []
    for i in range(len(lattice.nodes)):
        if isin(lattice.nodes[i,:2], shape, sx, sy): I_ok.append(I[i])
        
    nodes = nodes[I_ok,:]
    # Redefining the parameters corresponding to the elems
    ratio = keep_elem_prop(elems, lattice.aspect_ratio, I_ok)
    if isinstance(lattice.beam_section, np.ndarray):
        section = keep_elem_prop(elems, lattice.beam_section, I_ok)
    else: section = lattice.beam_section

    elems = keep_elem(elems, I_ok)
    return Lattice(nodes, elems, length=lattice.L,
                   inimesh=lattice.type, beam_section=section,
                   aspect_ratio=ratio)




"""
Redraw
======

Various tools to modify existing lattices with mappings and addition/deletion 
of elements.
The Perlin noise is generated using the Perlin-numpy library tools created by
João Pedro Vasconcelos and Bruno Barufaldi.
"""

#%% Package importation

from copy import deepcopy
import numpy as np

import numba as nb
from scipy.interpolate import griddata

from .lattices import Lattice

__all__ = ['mapping', 'flat_bounds', 'add_elements']

#%% Small functions
@nb.njit
def interpolant2d(t: np.ndarray):
    return t*t*t*(t*(t*6 - 15) + 10)

@nb.njit
def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant2d):
    """Generate a 2D numpy array of perlin noise.
    
    Parameters
    ----------
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
            
    Returns
    -------
        A numpy array of shape shape with the generated noise.
        
    Raises
    ------
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    xvals = np.arange(0,res[0], delta[0])
    yvals = np.arange(0,res[1], delta[1])
    grid = np.empty((2, len(xvals), len(yvals)))
    for j, y in enumerate(yvals):
        for i, x in enumerate(xvals):
            grid[0][i, j] = x

    for i, x in enumerate(xvals):
        grid[1][i, :] = yvals
    grid = grid.transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    grad_matrix = np.empty((d[0] * gradients.shape[0], 
                            d[1] * gradients.shape[1], 2))
    for i in range(gradients.shape[0]):
        for j in range(gradients.shape[1]):
            grad_matrix[i * d[0] : (i+1) * d[0], 
                        j * d[1] : (j+1) * d[1]] = gradients[i, j]
    gradients = grad_matrix
        
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    t1 = (1-t[:,:,1])*n0
    t2 = t[:,:,1]*n1
    sum_t = t1 + t2
    mult_s2 = np.sqrt(2)*sum_t
    return mult_s2

@nb.njit(parallel=True, fastmath=True)
def interpolant3d(t):
    return t ** 3 * (t * (t * 6 - 15) + 10)


@nb.njit(parallel=True, fastmath=True)
def generate_perlin_noise_3d(shape, res, tileable=(False, False, False)):
    """
    Generate 3D Perlin noise.
    
    References
    ----------
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin3d.py
    and
    https://github.com/pvigier/perlin-numpy/issues/9#issue-968667149

    """
    dtype = np.float32
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    range1 = np.arange(0, res[0], delta[0]).astype(dtype) % 1
    range2 = np.arange(0, res[1], delta[1]).astype(dtype) % 1
    range3 = np.arange(0, res[2], delta[2]).astype(dtype) % 1

    grid = np.empty(shape=(shape[0], shape[1], shape[2], 3), dtype=dtype)
    # grid -> [shape[0], shape[1], shape[2], 3]

    for idx in nb.prange(shape[0]):
        grid[idx, :, :, 0] = range1[idx]

    for idx in nb.prange(shape[1]):
        grid[:, idx, :, 1] = range2[idx]

    for idx in nb.prange(shape[2]):
        grid[:, :, idx, 2] = range3[idx]

    # Gradients
    theta = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)
    phi = 2 * np.pi * \
        np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1).astype(dtype)

    gradients = np.stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)),
        axis=-1
    )
    # gradients -> [res[0] + 1, res[1] + 1, res[2] + 1, 3]

    if tileable[0]:
        gradients[-1, :, :] = gradients[0, :, :]
    if tileable[1]:
        gradients[:, -1, :] = gradients[:, 0, :]
    if tileable[2]:
        gradients[:, :, -1] = gradients[:, :, 0]

    grad_shape = (
        d[0] * gradients.shape[0], d[1] * gradients.shape[1],
        d[2] * gradients.shape[2], 3)
    grad_matrix = np.empty(shape=grad_shape, dtype=dtype)

    for idx1 in nb.prange(gradients.shape[0]):
        for idx2 in nb.prange(gradients.shape[1]):
            for idx3 in nb.prange(gradients.shape[2]):
                grad_matrix[
                    d[0] * idx1: d[0] * (idx1 + 1),
                    d[1] * idx2: d[1] * (idx2 + 1),
                    d[2] * idx3: d[2] * (idx3 + 1),
                ] = gradients[idx1, idx2, idx3]

    gradients = grad_matrix
    # gradients -> [shape[0] + d[0], shape[1] + d[1], shape[2] + d[2], 3]

    g000 = gradients[:-d[0], :-d[1], :-d[2]]
    g100 = gradients[d[0]:, :-d[1], :-d[2]]
    g010 = gradients[:-d[0], d[1]:, :-d[2]]
    g110 = gradients[d[0]:, d[1]:, :-d[2]]
    g001 = gradients[:-d[0], :-d[1], d[2]:]
    g101 = gradients[d[0]:, :-d[1], d[2]:]
    g011 = gradients[:-d[0], d[1]:, d[2]:]
    g111 = gradients[d[0]:, d[1]:, d[2]:]
    # gxy -> [shape[0], shape[1], shape[2], 3]

    # Ramps

    n_bits = 3
    len_ = 2 ** n_bits
    code = ((np.arange(len_).reshape(len_, 1) & (1 << np.arange(n_bits)))) > 0
    code = code.astype(np.int32)
    # gradients -> [8, 3]

    n000 = np.sum((grid - code[0]) * g000, 3)
    n100 = np.sum((grid - code[1]) * g100, 3)
    n010 = np.sum((grid - code[2]) * g010, 3)
    n110 = np.sum((grid - code[3]) * g110, 3)
    n001 = np.sum((grid - code[4]) * g001, 3)
    n101 = np.sum((grid - code[5]) * g101, 3)
    n011 = np.sum((grid - code[6]) * g011, 3)
    n111 = np.sum((grid - code[7]) * g111, 3)
    # nxyz -> [shape[0], shape[1], shape[2]]

    t = interpolant3d(grid)
    t1 = 1 - t[:, :, :, 0]

    n00 = t1 * n000 + t[:, :, :, 0] * n100
    n10 = t1 * n010 + t[:, :, :, 0] * n110
    n01 = t1 * n001 + t[:, :, :, 0] * n101
    n11 = t1 * n011 + t[:, :, :, 0] * n111

    t2 = 1 - t[:, :, :, 1]
    n0 = t2 * n00 + t[:, :, :, 1] * n10
    n1 = t2 * n01 + t[:, :, :, 1] * n11

    output = (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1

    return output


#%% Main functions

def call_fun(func, fun_param, shape, dim):
    if func=='perlin':
        rep = fun_param[:dim]
        shape = rep*np.ceil(shape[:dim]/rep)
        tile = (True,True,False)[:dim]
        
        if dim==3:
            return generate_perlin_noise_3d(
                [shape[1], shape[0], shape[2]],
                res=[rep[1], rep[0], rep[2]],
                tileable=tile)
        else:
            return generate_perlin_noise_2d(
                [shape[1], shape[0]], res=[rep[1],rep[0]], tileable=tile
                )
        
    if func=='rayleigh':
        rng = np.random.default_rng()
        res = rng.rayleigh(scale=fun_param, size=shape)
    
    if func=='chi2':
        rng = np.random.default_rng()
        res = rng.chisquare(fun_param, shape)
        
    if func=='exp':
        rng = np.random.default_rng()
        res = rng.exponential(fun_param, shape)
        
    if func=='gamma':
        rng = np.random.default_rng()
        res = rng.gamma(fun_param[0], scale=fun_param[1], size=shape)
        
    if func=='gumbel':
        rng = np.random.default_rng()
        res = rng.gumbel(fun_param[0], scale=fun_param[1], size=shape)
        
    if func=='laplace':
        rng = np.random.default_rng()
        res = rng.laplace(fun_param[0], scale=fun_param[1], size=shape)
    
    if func=='logistic':
        rng = np.random.default_rng()
        res = rng.logistic(fun_param[0], scale=fun_param[1], size=shape)
        
    if func=='lognormal':
        rng = np.random.default_rng()
        res = rng.lognormal(fun_param[0], sigma=fun_param[1], size=shape)
        
    if func=='normal':
        rng = np.random.default_rng()
        res = rng.normal(fun_param[0], scale=fun_param[1], size=shape)
        
    if func=='pareto':
        rng = np.random.default_rng()
        res = rng.pareto(fun_param, shape)
        
    if func=='poisson':
        rng = np.random.default_rng()
        res = rng.poisson(fun_param, shape)
        
    if func=='lorentz':
        rng = np.random.default_rng()
        res = rng.standard_cauchy(shape)
        
    if func=='student':
        rng = np.random.default_rng()
        res = rng.standard_t(fun_param, shape)
        
    if func=='vonmises':
        rng = np.random.default_rng()
        res = rng.vonmises(fun_param[0], fun_param[1], size=shape)
        
    if func=='wald':
        rng = np.random.default_rng()
        res = rng.wald(fun_param[0], fun_param[1], size=shape)
        
    if func=='weibull':
        rng = np.random.default_rng()
        res = rng.weibull(fun_param, shape)
        
    if func=='zipf':
        rng = np.random.default_rng()
        res = rng.zipf(fun_param, shape)
        
    res = res.astype(np.float64)
    res-=np.min(res)
    res/=np.max(res)
    
    return res
    

def mapping(lattice: Lattice, func: str|np.ndarray, fun_param = None,
            where: str = 'elems', return_pos: bool = False):
    """
    Create a mapping from a given distribution over the nodes or elements.

    Parameters
    ----------
    lattice : Lattice
        Lattice over which the mapping is done.
    func : str or np.ndarray
        If str: name of the mapping function used. Ditributions implemented 
        in treillis are:
        
        - '*chi2*' Chi-square
        - '*exp*' Exponential
        - '*gamma*' Gamma
        - '*gumbel*' Gumbel
        - '*laplace*' Laplace
        - '*logistic*' Logistic
        - '*lognormal*' log-normal
        - '*lorentz*' Lorentz
        - '*normal*' normal
        - '*pareto*' Pareto
        - '*perlin*' Perlin noise (based on `numpy-perlin`, written by
                                   J. P. Vasconcelos and B. Barufaldi)
        - '*poisson*' Poisson
        - '*rayleigh*' Rayleigh
        - '*student*' Student
        - '*vonmises*' Von Mises
        - '*wald*' Wald
        - '*weibull*' Weibull
        - '*zipf*' Zipf
            
    fun_param : depends on the distribution
        Control parameter of the distribution apart from `scale`, which is 
        defined by the lattice shape. For the numpy functions, see the numpy documentation. For Perlin noise, tuple corresponding to the number of repetitions of the pattern along each axe.
    where : str, 'nodes' or 'elems'
        Whether to map over the nodes or the elements of the lattice. The
        default is 'elems'.
    return_pos : bool, optional
        Return the position in space of each point in the mapping. The
        default is False.
        
    See Also
    --------
    numpy.random.Generator.chisquare
    numpy.random.Generator.exponential
    numpy.random.Generator.gamma
    numpy.random.Generator.gumbel
    numpy.random.Generator.laplace
    numpy.random.Generator.logistic
    numpy.random.Generator.lognormal
    numpy.random.Generator.standard_cauchy
    numpy.random.Generator.normal
    numpy.random.Generator.pareto
    numpy.random.Generator.poisson
    numpy.random.Generator.rayleigh
    numpy.random.Generator.standard_t
    numpy.random.Generator.vonmises
    numpy.random.Generator.wald
    numpy.random.Generator.weibull
    numpy.random.Generator.zipf


    Returns
    -------
    list[mapping, positions] or mapping
    
    Examples
    --------
    Mapping the lattice elements to change the aspect ratio.
    
    >>> import treillis
    >>> from treillis.display import fast as fd
    >>> lat = treillis.loadLattice('lattice_130')
    >>> mapped, pos = treillis.mapping(lat, 'poisson', 5, return_pos=True)
    >>> fd.scatter_lat(lat, pos, mapped)
    >>> # Changing the aspect ratio based on the mapping
    >>> mapped -= mapped.mean()
    >>> lat.aspect_ratio *= 1 + mapped
    >>> fd.previz(lat)
    """
    if where not in ['nodes', 'elems']:
        raise ValueError('You can only map over the nodes or the elems.')
    
    size = np.ptp(lattice.nodes, axis=0)
    shape = np.ceil(2*size / lattice.L + 1).astype(int)
    dim = lattice.dim
    if dim==2: 
        shape = np.array([shape[0],shape[1], 0])
        size = np.array([size[0], size[1], 0])

    if isinstance(func, str):
        noise = call_fun(func, fun_param, shape, dim)
    else: noise = func

    if where == 'elems':
        position = (lattice.nodes[lattice.elems[:,1].astype(int),:]\
            + lattice.nodes[lattice.elems[:,0].astype(int),:]) / 2
    elif where == 'nodes': position = lattice.nodes

    x, y, z = np.indices(shape, dtype=float)
    x *= size[0]/shape[0].astype(int)
    y *= size[1]/shape[1].astype(int)
    z *= size[2]/shape[2].astype(int)
    xyz = np.concatenate((
        x.flatten().reshape(np.prod(x.shape),1),
        y.flatten().reshape(np.prod(y.shape),1),
        z.flatten().reshape(np.prod(z.shape),1)), axis=1, dtype=float)
        
    xyz = xyz - np.repeat(size.astype(float).reshape(1,dim)/2, xyz.shape[0],
                     axis=0)
    
    xyz = xyz[:,:dim]

    data = noise.flatten()

    ip = griddata(xyz, data, 
                       position,
                       method='nearest')
    
    ip -= np.min(ip, axis=None)
    ip /= np.max(ip, axis=None)
    
    if return_pos: return ip, position
    else: return ip.ravel()
    

def flat_bounds(lattice: Lattice, in_place=False):
    """Flatten the boundaries of the lattices by putting the fixed_nodes at an average position."""
    if not in_place: lattice = deepcopy(lattice)
    lattice.clean_lattice()
    fixed = np.array(lattice.fixed_nodes)
    
    if lattice.domain == 'rectangle':
        maxleft = np.min(lattice.nodes[:,0])
        left = np.flatnonzero(lattice.nodes[fixed,0]-maxleft
                              <= lattice.L).astype(int)
        left = fixed[left.astype(int)]
        meanleft = np.mean(lattice.nodes[left,0])
        
        lattice.nodes[left,0] = meanleft
        
        maxright = np.max(lattice.nodes[:,0])
        right = np.flatnonzero(maxright - lattice.nodes[fixed,0]
                               <= lattice.L).astype(int)
        right = fixed[right]
        meanright = np.mean(lattice.nodes[right,0])
        
        lattice.nodes[right,0] = meanright
        
        maxdown = np.min(lattice.nodes[:,1])
        down = np.flatnonzero(lattice.nodes[fixed,1]-maxdown
                              <= lattice.L).astype(int)
        down = fixed[down]
        meandown = np.mean(lattice.nodes[down,1])
        
        lattice.nodes[down,1] = meandown
        
        maxup = np.max(lattice.nodes[:,1])
        up = np.flatnonzero(maxup - lattice.nodes[fixed,1] 
                            <= lattice.L).astype(int)
        up = fixed[up]
        meanup = np.mean(lattice.nodes[up,1])
        
        lattice.nodes[up,1] = meanup
        
        elm_add = np.array([[],[]]).T
        keep_f = np.zeros_like(fixed, dtype=bool)
        
        # Putting new elements on the edges to mark the shape
        if lattice.dim == 2:
            for l, side in zip([left, right, down, up], [1,1,0,0]):
                l = l[np.argsort(lattice.nodes[l,side])]
                elm_add = np.concatenate(
                    (elm_add, np.stack((l, np.roll(l, -1)), axis=-1)[:-1]), 
                    axis=0)
                keep_f += np.isin(fixed, l)
            

            lattice._fixed_nodes = lattice._fixed_nodes[keep_f]
            lattice.elems = np.concatenate((lattice.elems[:,:2], elm_add),
                                           axis=0)
            lattice.elems = np.unique(lattice.elems, axis=0)
            
            lattice.elems = np.concatenate(
                (lattice.elems, lattice.length[:,None]), axis=1)
            
        
        if lattice.dim == 3:
            maxback = np.min(lattice.nodes[:,2])
            back = np.flatnonzero(lattice.nodes[fixed,2]-maxback
                                  <= lattice.L).astype(int)
            back = fixed[back]
            meanback = np.mean(lattice.nodes[back,2])
            
            lattice.nodes[back,2] = meanback
            
            maxfront = np.max(lattice.nodes[:,2])
            front = np.flatnonzero(maxfront\
                                   - lattice.nodes[fixed,2] 
                                   <= lattice.L).astype(int)
            front = fixed[front]
            meanfront = np.mean(lattice.nodes[front,2])
            
            lattice.nodes[front,2] = meanfront
            
            couple = [(left, front), (front, right), (right, back),
                       (back, left), (up, front), (front, down), (down, back),
                       (back, up), (left, down), (down, right), (right, up),
                       (up, left)]
            side = [2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1]
            
            for c, s in zip(couple, side):
                l = c[0][np.isin(c[0], c[1])]
                l = l[np.argsort(lattice.nodes[l, s])]
                elm_add = np.concatenate(
                    (elm_add, np.stack((l, np.roll(l, -1)), axis=-1)[:-1]), 
                    axis=0)
                lattice.elems = np.concatenate((lattice.elems[:,:2], elm_add),
                                               axis=0)
                lattice.elems = np.unique(lattice.elems, axis=0)
                lattice.elems = np.concatenate(
                    (lattice.elems, lattice.length[:,None]), axis=1)
        
    elif lattice.domain == 'ball':
        norm = np.sqrt(np.sum(lattice.nodes[fixed]**2, axis=1))
        Rmean = np.mean(norm)
        
        vec = lattice.nodes[fixed] / np.repeat(
            norm.reshape(len(norm),1), lattice.dim, axis=1)
        
        lattice.nodes[fixed] = Rmean * vec
        elm_add = np.array([[],[]]).T
        
        if lattice.dim == 2:
            angle = np.atan2(lattice.nodes[fixed,1], lattice.nodes[fixed,0])
            fixed = fixed[np.argsort(angle)]
            elm_add = np.stack((fixed, np.roll(fixed, -1)), axis=-1)[:-1]
            lattice.elems = np.concatenate((lattice.elems[:,:2], elm_add),
                                           axis=0)
            lattice.elems = np.unique(lattice.elems, axis=0)
            lattice.elems = np.concatenate(
                (lattice.elems, lattice.length[:,None]), axis=1)
            
    else:
        raise AttributeError("Only rectangle- and ball-shaped lattices can be flattened.")
   
    lattice.aspect_ratio = np.concatenate(
        (lattice.aspect_ratio, np.full(
            lattice.elems.shape[0]-lattice.aspect_ratio.shape[0],
            lattice.aspect_ratio.mean())))
        
    if isinstance(lattice.beam_section, np.ndarray):
        lattice.beam_section = np.concatenate(
            (lattice.beam_section, np.full((lattice.elems.shape[0]
                                           - lattice.aspect_ratio.shape[0],
                                           lattice.beam_section.shape[1],
                                           lattice.beam_section.shape[2]),
                                           lattice.beam_section.mean(axis=0))))

    if np.flatnonzero(lattice.length>lattice.L*3).shape[0] > 0:
        lattice.del_elm(lattice.length>lattice.L*3)
        
    # # Creating new nodes at element intersection
    # for i in lattice.fixed_nodes:
    #     D = np.sqrt(np.sum(
    #         (lattice.nodes[lattice.fixed_nodes]-lattice.nodes[i][None,:])**2, 
    #         axis=1))
    #     neighb = lattice.fixed_nodes[D < 2*lattice.L]
    #     elmi = 
    return lattice


def add_elements(lattice: Lattice, start_points: np.ndarray, 
                 end_points: np.ndarray, in_place: bool = True):
    """
    Add new elements to a lattice.
    
    New elements are entered as tables of initial and final points, and are
    recalculated so that they stop at the first element they cross from the
    initial lattice.

    Parameters
    ----------
    lattice : Lattice
        Lattice to add elements to.
    start_points : (Nelems, Ndim) np.ndarray
        Position of the initial points for the new elements. Positions will
        be recalculated from the closest point on the original lattice.
    end_points : (Nelems, Ndim) np.ndarray
        Positions of the final points of the new elements. Positions will be
        recalculated to stopp at the first crossing.
    in_place : bool, optional
        Change the lattice in place or create a copy. The default is True.

    Returns
    -------
    Lattice
    """
    
    if not in_place: lattice = deepcopy(lattice)
    
    add_elems = np.zeros((start_points.shape[0], 3))
    
    for i in range(add_elems.shape[0]):
        # Get starting points by finding points in the lattice closest
        # to the given points by user
        pt = start_points[i]
        roi = np.flatnonzero(np.sum((lattice.nodes - pt)**2, axis=1)
                            < 2* lattice.L)
        Ielms = np.flatnonzero(np.sum(np.isin(lattice.elems[:,:2], roi),
                   axis=1, dtype=bool))
        elms = lattice.elems[Ielms,:2].astype(int)
        vecs = lattice.nodes[elms[:,1]] - lattice.nodes[elms[:,0]]
        
        if lattice.dim == 2:
            # Find point in the lattices closest to the start points
            m = (lattice.nodes[elms[:,1],1]
                 - lattice.nodes[elms[:,0],1])\
                / (lattice.nodes[elms[:,1],0]
                     - lattice.nodes[elms[:,0],0])
            b = lattice.nodes[elms[:,0],1]\
                - m*lattice.nodes[elms[:,0],0]
                
            ox = (vecs[:,0]*pt[0] + vecs[:,1]*pt[1] - vecs[:,1]**2)\
                / (vecs[:,0] + m*b)
            cross_pt = np.stack((ox, m*ox + b), axis=1)
            cross_v = cross_pt - lattice.nodes[elms[:,0]]
            ratio = np.nanmean(cross_v / vecs, axis=1)
            cross_pt = cross_pt[(ratio>0) & (ratio<1)]
            Ielms = Ielms[(ratio>0) & (ratio<1)]
        
        if lattice.dim == 3:
            lmbda = (vecs[:,0]*pt[0] + vecs[:,1]*pt[1] + vecs[:,2]*pt[2])\
                / (vecs[:,0]**2 + vecs[:,1]**2 + vecs[:,2]**2)
            Ielms = Ielms[(lmbda>0) & (lmbda<1)]
            vecs = vecs[(lmbda>0) & (lmbda<1)]
            lmbda = lmbda[(lmbda>0) & (lmbda<1)]
            cross_pt = lmbda[:,None] * vecs\
                + lattice.nodes[lattice.elems[Ielms,0].astype(int)]
            # print(lmbda)
            
        d = np.sqrt(np.sum((pt - cross_pt)**2, axis=1))
        
        d_nodes = np.sqrt(np.sum((pt - lattice.nodes[roi])**2, 
                                 axis=1))
        # Find whether the new element starts on an already existing node
        if len(d)>0:
            who = np.argmin([np.min(d), np.min(d_nodes)])
        else: who = 1
        
        if who == 0:
            lattice.nodes = np.concatenate(
                (lattice.nodes, cross_pt[np.argmin(d)][None,:]), axis=0)
            # Add the new node in the element
            add_elems[i,0] = lattice.nodes.shape[0] - 1
            elm_change = lattice.elems[Ielms[np.argmin(d)]]
            lattice.elems =  np.delete(lattice.elems, Ielms[np.argmin(d)],
                                       axis=0)
            lattice.elems = np.concatenate(
                (lattice.elems,
                 [[elm_change[0], lattice.nodes.shape[0] - 1,0]],
                 [[elm_change[1], lattice.nodes.shape[0] - 1,0]]),
                axis=0)
        else:
            # Simply add the node that will have an additionnal neighbor to
            # the table
            add_elems[i,0] = roi[np.argmin(d_nodes)]
            
    # Get the end nodes
    for i in range(add_elems.shape[0]):
        # Again, keep the closest nodes
        pt = (start_points[i] + end_points[i]) / 2
        roi = np.flatnonzero(np.sum((lattice.nodes - pt)**2, axis=1)
                            < 5 * lattice.L)
        Ielms = np.flatnonzero(np.sum(np.isin(lattice.elems[:,:2], roi),
                   axis=1, dtype=bool))
        elms = lattice.elems[Ielms,:2].astype(int)
        vecs = lattice.nodes[elms[:,1]] - lattice.nodes[elms[:,0]]
        x = np.zeros((2, len(Ielms)))
        
        # Calculate the intersection of the segments
        for ie in range(len(Ielms)):
            A = np.zeros((lattice.dim, 2))
            A[:,0] = start_points[i] - end_points[i]
            A[:,1] = vecs[ie]
            
            B = start_points[i] - lattice.nodes[elms[ie,0]]
            try:
                xy = np.linalg.lstsq(A, B)
                x[:,ie] = xy[0]
                if B[0]==0 and B[1]==0:
                    if lattice.dim==2: x[:,ie] = 1000
                    elif B[2]==0: x[:,ie] = 1000
            except: x[:,ie] = 1000

        # Find the intersection point closest to start
        inseg = np.prod((x>=0)&(x<=1), axis=0, dtype=bool)
        pts = lattice.nodes[elms[inseg,0]] + x[1,inseg][:,None] * vecs[inseg]
        which = np.argmin(np.sqrt(np.sum((pts - start_points[i])**2, 
                                         axis=1)))
        # If intersection is at an already existing node, just add it to 
        # new elements
        if x[1,inseg][which] in [0,1]:
            add_elems[i,1] = elms[np.flatnonzero(inseg)[which],
                                  int(x[1,inseg][which])]
        else:
            lattice.nodes = np.concatenate(
                (lattice.nodes, pts[which][None,:]), axis=0)
            # Add the new node in the element
            add_elems[i,1] = lattice.nodes.shape[0] - 1
            elm_change = lattice.elems[Ielms[inseg][which]]
            lattice.elems =  np.delete(lattice.elems, Ielms[inseg][which],
                                       axis=0)
            lattice.elems = np.concatenate(
                (lattice.elems,
                 [[elm_change[0], lattice.nodes.shape[0] - 1,0]],
                 [[elm_change[1], lattice.nodes.shape[0] - 1,0]]),
                axis=0)
                        
        lattice.elems = np.concatenate((lattice.elems, add_elems[i][None,:]),
                                       axis=0)
        
        lattice.length
        lattice.elems = np.unique(lattice.elems, axis=0)

        # Update the element depending values
        lattice.aspect_ratio = np.concatenate((
            lattice.aspect_ratio, [lattice.aspect_ratio.mean()] * (
                lattice.elems.shape[0]-lattice.aspect_ratio.shape[0])))
                                  
    return lattice
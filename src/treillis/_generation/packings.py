"""
Packings
========

Create iostropic meshes from a random close pckg of solid spheres with controlled radius.

The packing software was created by Vasili Baranov, and is available `on Github`_.

.. _on Github: https://github.com/VasiliBaranov/pckg-generation
"""


# %% Package importation
import os
from itertools import combinations
import numpy as np
from scipy.spatial import Delaunay, Voronoi
import pandas as pd
from numba import njit, jit, prange
import ipyparallel as ipp
from time import time

__all__ = ["makepacking", "frompacking"]

TYPEFLOAT = np.float64
TYPEVOR = np.float32 # Voronoi calculations are too costly to allow for float64

PREPACKPATH = os.path.abspath(os.path.dirname(__file__) + r'/prepacking/')
PACKPATH = os.path.abspath(os.path.dirname(__file__)
                           + r'/3D-packing-generation/')
PACK2DPATH = os.path.abspath(os.path.dirname(__file__) 
                             + r'/2D-packing-generation/')
# %% Useful fonctions

def delete_dupl_nodes(nodes, elems, length):
    """
    Delete node that are too close to each other.

    Nodes are considered duplicates when the distance is inferior to
    0.0001*length. Process is sped up by the hypothesis that close neighbors
    are linked together which could lead to errors in less than 1 in 10 cases.

    Parameters
    ----------
    nodes : ARRAY OF FLOATS
        nodes array.
    length : FLOAT
        size of the mesh

    Returns
    -------
    nodes : ARRAY OF FLOATS
        array of kept unique nodes.

    """
    Del = [] # list of nodes to delete

    eps = length * 1e-4

    L = np.sqrt(np.sum( (nodes[elems[:,1].astype(int)]
                         -nodes[elems[:,0].astype(int)])**2, axis=1 ))

    if len(L[L<=eps])>0:
        Del = np.unique(elems[L<=eps,1])
        print('Deleting duplicates: ', Del)
        nodes = np.delete(nodes, Del.astype(int), axis=0)
    return nodes


@jit(nopython=True, fastmath=True)
def dist_jit(x, y, length):
    """
    Check if distance between 2 nodes is inferior to 0.0001*length.

    Parameters
    ----------
    x : ARRAY OF FLOATS
        first node.
    y : ARRAY OF FLOATS
        second node.
    length : FLOAT
        average element's length before disorder.

    Returns
    -------
    BOOL
        True if the distance is inferior to 0.0001*length.

    """
    EPS = 1E-04 * length
    return np.sqrt(np.sum((x - y)**2, axis=1)) < EPS


def find_elems_inside(size, nodes, elems):
    """
    Find the elements inside the domain.

    Accepts both elems encoded as indexes of the nodes' table or directly
    encoded as coordinates (auto-detected).

    Parameters
    ----------
    size : ARRAY OF FLOATS
        size of the domain.
    nodes : ARRAY OF FLOATS
        nodes array.
    elems : ARRAY OF FLOATS
        elements array.

    Returns
    -------
    elems : ARRAY OF FLOATS
        coordinates of the elements that are inside the domain.
    elem_kept1 : ARRAY OF BOOL
        test if the first node in elems is inside the domain.
    elem_kept2 : ARRAY OF BOOL
        test if the second node in elems is inside the domain.

    """
    EPS = 1E-04
    dim = np.shape(nodes)[1]

    # auto-detect how the elements are encoded, and retrieve the coordinates
    if len(np.shape(elems)) == 2:
        el1 = nodes[elems[:, 0], :]
        el2 = nodes[elems[:, 1], :]
    else:
        el1 = elems[:, 0, :]
        el2 = elems[:, 1, :]

    # following the domain type find which elements are inside:
    if len(size) == 1: # circle or sphere
        elem_kept1 = np.sqrt(np.sum(el1 ** 2, axis=1)) <= size[0] + EPS
        elem_kept2 = np.sqrt(np.sum(el2 ** 2, axis=1)) <= size[0] + EPS

    if len(size) == 2 or len(size) == 3: # rectangle or brick
        elem_kept1 = np.all(np.abs(el1) <= size / 2 + EPS, axis=1)
        elem_kept2 = np.all(np.abs(el2) <= size / 2 + EPS, axis=1)

    if len(size) == 5: # wedge
        domain_subtestright_elm1 = (el1[:, 1] > -size[1] / 2 - EPS)\
            * (el1[:, 1] < size[1] / 2 + EPS)\
            *  (el1[:, 0] > (size[4] - size[0] / 2) - EPS)\
            *  (el1[:, 0] < size[0] / 2 + EPS)

        elem_kept1 = (el1[:, 0] > -size[0] / 2 - EPS)\
            * (el1[:, 0] < (size[4] - size[0] / 2) - EPS)\
                * (el1[:, 1] > -size[1] / 2 - EPS)\
                    * (el1[:, 1] < size[1] / 2 + EPS)\
                        * ((el1[:, 1] < -size[3] / 2 + EPS)\
                           + (el1[:, 1] > size[3] / 2 - EPS))

        elem_kept1 = elem_kept1 + domain_subtestright_elm1

        domain_subtestright_elm2 = (el2[:, 1] > -size[1] / 2 - EPS)\
            * (el2[:, 1] < size[1] / 2 + EPS)\
                * (el2[:, 0] > (size[4] - size[0] / 2) - EPS)\
                    * (el2[:, 0] < size[0] / 2 + EPS)

        elem_kept2 = (el2[:, 0] > -size[0] / 2 - EPS)\
            * (el2[:, 0] < (size[4] - size[0] / 2) - EPS)\
                * (el2[:, 1] > -size[1] / 2 - EPS)\
                    * (el2[:, 1] < size[1] / 2 + EPS)\
                        * ((el2[:, 1] < -size[3] / 2 + EPS)\
                           + (el2[:, 1] > size[3] / 2 - EPS))

        elem_kept2 = elem_kept2 + domain_subtestright_elm2

        if dim ==3:
            elem_kept1 = elem_kept1 * (np.abs(el1[:, 2]) < size[2] / 2 + EPS)
            elem_kept2 = elem_kept2 * (np.abs(el2[:, 2]) < size[2] / 2 + EPS)

    elem_kept = elem_kept1 * elem_kept2

    if len(np.shape(elems)) == 2:
        elems = nodes[elems[elem_kept, :]]
    else:
        elems = elems[elem_kept, :, :]

    return elems, elem_kept1, elem_kept2

@njit(parallel=True)
def find_nodes_bound(size, nodes, elems, elem_kept1, elem_kept2):
    """
    Find the boundary nodes.

    The boundary nodes are linked to nodes outside the domain. Accept both
    elems encoded as indexes of the nodes table or directly encoded as
    coordinates (auto-detected).

    Parameters
    ----------
    size : ARRAY OF FLOATS
        size of the domain.
    nodes : ARRAY OF FLOATS
        nodes array.
    elems : ARRAY OF FLOATS
        elems array.
    elem_kept1 : ARRAY OF BOOL
        test if the first node in elems is inside the domain
        (returns of find_elems_inside).
    elem_kept2 : ARRAY OF BOOL
        test if the second node in elems is inside the domain
        (returns of find_elems_inside).

    Returns
    -------
    fixed : ARRAY OF FLOATS
        coordinates of the boundary nodes.

    """
    EPS = 1E-04
    dim = np.shape(nodes)[1]
    elems_boundary = np.flatnonzero(elem_kept1 + elem_kept2 == 1)
    fixed = np.zeros((len(elems_boundary), dim), dtype=nodes.dtype)

    # following the domain type find which nodes are on the interior boundary:
    for k in range(len(elems_boundary)):
        if len(size) == 1:
            var1 = np.sqrt(np.sum(elems[elems_boundary[k], 0, :] ** 2))
            fixed[k, :] = elems[elems_boundary[k],
                                int(not(var1 <= size[0] + EPS)), :]

        if len(size) == 2 or len(size) == 3:
            var1 = np.prod(np.abs(elems[elems_boundary[k], 0, :])
                           <= size / 2 + EPS)
            fixed[k, :] = elems[elems_boundary[k], int(not(var1)), :]

        if len(size) == 5:
            domain_subtestright_fixed = (elems[elems_boundary[k], 0, 1]
                                         > -size[1] / 2 - EPS)\
                and (elems[elems_boundary[k], 0, 1] < size[1] / 2 + EPS)\
                    and (elems[elems_boundary[k], 0, 0]
                         > (size[4] - size[0] / 2) - EPS)\
                        and (elems[elems_boundary[k], 0, 0]
                             < size[0] / 2 + EPS)

            var1 = (elems[elems_boundary[k], 0, 0] > -size[0] / 2 - EPS)\
                and (elems[elems_boundary[k], 0, 0]
                     < (size[4] - size[0] / 2) - EPS)\
                    and (elems[elems_boundary[k], 0, 1] > -size[1] / 2 - EPS)\
                        and (elems[elems_boundary[k], 0, 1] < size[1] / 2 + EPS)\
                            and ((elems[elems_boundary[k], 0, 1]
                                  < -size[3] / 2 + EPS)\
                                 or (elems[elems_boundary[k], 0, 1]
                                     > size[3] / 2 - EPS))

            var1 = var1 or domain_subtestright_fixed

            if dim == 3:
                var1 = var1 and (np.abs(elems[elems_boundary[k], 0, 2])
                                 < size[2] / 2 + EPS)

            fixed[k, :] = elems[elems_boundary[k], int(not(var1)), :]
    return fixed


def find_nodes_in(size, nodes):
    """
    Find the nodes inside the domain.

    Parameters
    ----------
    size : ARRAY OF FLOATS
        size of the domain.
    nodes : ARRAY OF FLOATS
        nodes array.

    Returns
    -------
    nodes : ARRAY OF FLOATS
        array of the coordinates of the nodes inside the domain.

    """
    EPS = 1E-04
    dim = np.shape(nodes)[1]

    if len(size) == 1:
        nodes_kept = np.sqrt(np.sum(nodes ** 2, axis=1)) <= size[0] + EPS

    if len(size) == 2 or len(size) == 3:
        nodes_kept = np.all((np.abs(nodes)
                             <= size.reshape((1, len(size))) / 2+EPS), axis=1)

    if len(size) == 5:
        domain_subtestright = (nodes[:, 1] > -size[1] / 2 - EPS)\
            * (nodes[:, 1] < size[1] / 2 + EPS)\
                * (nodes[:, 0] > (size[4] - size[0] / 2) - EPS)\
                    * (nodes[:, 0] < size[0] / 2 + EPS)

        nodes_kept = (nodes[:, 0] > -size[0] / 2 - EPS)\
            * (nodes[:, 0] < (size[4] - size[0] / 2) - EPS)\
                * (nodes[:, 1] > -size[1] / 2 - EPS)\
                    * (nodes[:, 1] < size[1] / 2 + EPS)\
                        * ((nodes[:, 1] < -size[3] / 2 + EPS)\
                           + (nodes[:, 1] > size[3] / 2 - EPS))

        nodes_kept = nodes_kept + domain_subtestright

        if dim ==3:
            nodes_kept = nodes_kept * (np.abs(nodes[:, 2]) < size[2] / 2 + EPS)

    nodes = nodes[nodes_kept, :]

    return nodes

def get_e(args):
    import numpy as np
    nodes, e, length = args
    e1 = np.flatnonzero(np.sqrt(np.sum((nodes-e[0])**2, axis=1))
                       < 1e-4 * length) 
    e2 = np.flatnonzero(np.sqrt(np.sum((nodes-e[1])**2, axis=1))
                       < 1e-4 * length)
    return e1[0], e2[0]

def find_indices(nodes, elems, fixed, length):
    """
    Find the new indexes of the elems.

    elems are encoded as nodes coordinates and this function returns elems
    encoded as indices in the nodes array.


    Parameters
    ----------
    nodes : ARRAY OF FLOATS
        ndoes array.
    elems : ARRAY OF FLOATS
        elems array.
    fixed : ARRAY OF FLOATS
        coordinates of the boundary nodes.
    length : FLOAT
        average beam length.

    Returns
    -------
    elems : ARRAY OF FLOATS
        indices in the nodes array of the elems.
    fixed : ARRAY OF FLOATS
        indices in the nodes array of the boundary nodes.

    """
    dim = np.shape(nodes)[1]
    
    if elems.shape[0] < 10000:
        e1 = np.array([
            np.flatnonzero(np.sqrt(np.sum((nodes-e[0])**2, axis=1))
                           < 1e-4 * length) 
            for e in elems]).ravel()
        e2 = np.array([
            np.flatnonzero(np.sqrt(np.sum((nodes-e[1])**2, axis=1))
                           < 1e-4 * length)
            for e in elems]).ravel()
        elems = np.stack((e1.ravel(), e2.ravel()), axis=-1)
    else:
        with ipp.Cluster(n=int(os.cpu_count()*2/3)) as rc:
            dview = rc[:]

            args = [[nodes, e, length] for e in elems]
            res = dview.map_async(get_e, args);
            res.wait_interactive()
        elems = np.stack(res)
    
    elems.sort(axis=1)
    elems = np.unique(elems, axis=0)
    elems = np.concatenate((elems, np.zeros((np.shape(elems)[0], 1),
                                            dtype=elems.dtype)), axis=1)

    f = np.zeros(fixed.shape[0])
    for k in range(np.shape(fixed)[0]):
        f[k] = np.flatnonzero(np.sqrt(np.sum((nodes-fixed[k])**2, axis=1))
                           <= length * 1e-4)[0]
    #     fixed[k, 0] = np.flatnonzero(dist_jit(fixed[k, :].\
    #                                       reshape((1, dim)), nodes, length))
    #     fixed[k, 1] = fixed[k, 0]
    #     if dim == 3:
    #         fixed[k, 2] = fixed[k, 0]

    # fixed = np.sum(fixed, axis=1) / dim
    fixed = np.unique(f)

    ret = (elems, fixed)
    return ret

def launch_pack(path, N_particles, side, *, algo='ls', start=1,
                            contraction_rate=1e-001, erase=False):
    """
    Launch the packing generator software from Vasili Baronov.

    Parameters
    ----------
    path : STR
        Path to the packing.
    N_particles : INT
        Number of beads in the packing.
    side : (3,) ARRAY OF FLOATS
        sizes of the packing box.
    algo : STR, optional
        packing algorithm. The default is 'ls'.
        Different options are:
        - 'ls': Lubachevsky–Stillinger
        - 'lsgd': Lubachevsky–Stillinger with gradual densification
        - 'lsebc': Lubachevsky–Stillinger with equilibration between compressions
        - 'fba': force-biased algorithm
        - 'ojt': original Jodrey–Tory algorithm
        - 'kjt':  Jodrey–Tory algorithm modification by Khirevich
        - 'mca': Monte Carlo algorithm
        - 'cja': Conjugate gradient. Requires GNU Scientific library and additional compiling options
    start : INT, optional
        Start a new simulation (1) or use previous packing (0). The default is 1.
    contraction_rate : FLOAT, optional
        Contraction rate. The default is 1e-001.
    erase : BOOL, optional
        Choose whether to erase the previous packing. The default is False.
        If True, start has to be 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    Source
    ------
    [project github](https://github.com/VasiliBaranov/packing-generation)
    """
    if erase:
        test = os.listdir(path)
        for item in test:
            if item.endswith(".txt") or item.endswith(".nfo") or\
                item == 'packing_init.xyzd' or\
                item == 'packing_prev.xyzd':
                os.remove(os.path.join(path, item))

    flag = 'error'
    attempt = 0
    while flag == 'error' and attempt < 5:
        sample_config = "Particles count: {}\n".format(N_particles) \
                        + "Packing size: {} {} {}\n".\
                            format(side[0], side[1], side[2]) \
                        + "Generation start: {}\n".format(int(start)) \
                        + "Seed: {}\n".format(
                            np.random.default_rng().integers(2147483647)
                            ) \
                        + "Steps to write: 100\n" \
                        + "Boundaries mode: 1\n" \
                        + "Contraction rate: {}\n".format(contraction_rate)\
                        + "1. boundaries mode: 1 - bulk; 2 - ellipse"\
                            +" (inscribed in XYZ box, Z is length of an ellipse);"\
                                +" 3 - rectangle\n" \
                        + "2. generationMode = 1 (Poisson, R) or 2 (Poisson in cells, S)"


        with open(path + r'\generation.conf', 'w') as f:
            f.write(sample_config)


        os.chdir(path)
        os.system(path + r'\PackingGeneration__v1.0__Windows_7_8_10__x32.exe -'\
                  + algo)

        if not(os.path.isfile(path + r'\packing.nfo')):
            attempt += 1
        else:
            flag = 'ok'

    if not(os.path.isfile(path + r'\packing.nfo')):
        raise ValueError("The program did not succeed to create the packing.")
    pass


def pack_2D(size, Radius, Rmean, mu, sigma, r_min, r_max):
    """RCP based on Lozano et al., 2016."""
    EPS = 1e-2
    inter = True
    while inter:
        seed_pos = np.random.default_rng().uniform(low=-2.5*Rmean, high=2.5*Rmean, 
                                                   size=(3,2))
        seed_rad = np.random.default_rng().lognormal(mu, sigma, size=3)
        
        inter = np.array([np.sqrt(np.sum((seed_pos[i[0]]-seed_pos[i[1]])**2))
                 < seed_rad[i[0]] + seed_rad[i[1]] 
                 for i in combinations([0,1,2], 2)]).any()
    front_queue = np.arange(3).tolist()
    assembly = np.concatenate((
        seed_pos, seed_rad.reshape(len(seed_rad),1)), axis=1)
    grid = np.arange(3)

    while len(front_queue)>0:
        # create the neighboring box
        current = seed_pos[front_queue[0]]
        r_new = np.random.default_rng().lognormal(mu, sigma)
        neighbors = grid[
            (abs(abs(seed_pos[:,0]-current[0])-r_new-r_max)<=seed_rad)
            + (abs(abs(seed_pos[:,1]-current[1])-r_new-r_max)<=seed_rad)
            ]
        
        # find the candidate points by trigonometry, at the intersection of 
        # the halo circles
        vec = seed_pos[neighbors]-current
        lv = np.sqrt(np.sum(vec**2, axis=1))
        R0 = seed_rad[front_queue[0]] + r_new
        R1 = seed_rad[neighbors] + r_new
        keep = np.arange(len(neighbors))[(lv>0) & (lv<R1+R0)]
        neighbors = neighbors[keep]
        if len(neighbors)>0:
            vec = vec[keep]
            R1 = R1[keep]
            lv = lv[keep]
            for i in range(vec.shape[0]):
                vec[i,:] /= lv[i]
            u = np.concatenate((
                vec[:,1].reshape(len(vec),1),
                -vec[:,0].reshape(len(vec),1)), axis=1)
            
            
            l = ((R0**2 - R1**2 + lv**2) / (2*lv)).reshape(len(lv),1)
            h = np.sqrt(R0**2 - l**2).reshape(len(l),1)
            l = np.repeat(l, 2, axis=1)
            # l += 10*EPS
            h = np.repeat(h, 2, axis=1)
            
            candidates = np.concatenate((
                (current + l*vec + h*u).reshape(len(vec),2),
                (current + l*vec - h*u).reshape(len(vec),2)), axis=0)
            candidates = np.unique(candidates, axis=0)
            
            # keep only the candidates within the limit and that do not intercept other
            # spheres
            candidates = candidates[np.sqrt(np.sum(candidates**2, axis=1))<Radius]
            D = np.zeros((candidates.shape[0], seed_pos.shape[0]))
            for i in range(candidates.shape[0]):
                D[i] = np.sqrt(np.sum((candidates[i]-seed_pos)**2, axis=1))\
                    >= (r_new + seed_rad) - EPS
            candidates = candidates[np.prod(D, axis=1, dtype=bool)]
        else:
            candidates = np.empty((0,))
        
        if candidates.shape[0]<1:
            front_queue = front_queue[1:]
            
        if candidates.shape[0]>0:
            seed_rad = np.concatenate((seed_rad, np.array([r_new])))
            full_c = [np.concatenate((
                seed_pos,
                c.reshape(1,2)), axis=0)
                for c in candidates]
            max_rad = [max(np.sqrt(np.sum(
                (f-np.mean(f, axis=0))**2, axis=1))) for f in full_c]
            best = np.argmin(np.array(max_rad))

            seed_pos = np.concatenate((
                seed_pos,
                candidates[best].reshape(1,2)), axis=0)
            front_queue.append(len(seed_rad)-1)
            grid = np.concatenate((grid, np.array([len(seed_rad)-1])))
            assembly = np.concatenate((
                assembly,
                np.concatenate((candidates[best], np.array([r_new]))).reshape(1,3)),
                axis=0)
            
    np.save(PACKPATH + r'\packing', assembly)
            
    return assembly[:,:2]

def addnodes2side(nodes, size, dim, length):
    """Add more nodes on sides of wedge-splitting geometry to ensure a good strength."""
    a = size[0]
    b = size[1]
    d = size[3]
    e = size[4]

    if dim == 2:
        var = np.arange(-b / 2, b / 2, length)
        var = var.reshape((len(var), 1))
        newnodes = np.concatenate((np.full(np.shape(var), a / 2), var), axis=1)

        var = np.arange(-b / 2, -d / 2, length)
        var = var.reshape((len(var), 1))
        var = np.concatenate((np.full(np.shape(var), -a / 2), var), axis=1)
        newnodes = np.concatenate((newnodes, var), axis=0)

        var = np.arange(-d / 2, d / 2, length)
        var = var.reshape((len(var), 1))
        var = np.concatenate((np.full(np.shape(var), e - a / 2), var), axis=1)
        newnodes = np.concatenate((newnodes, var), axis=0)

        var = np.arange(d / 2, b / 2, length)
        var = var.reshape((len(var), 1))
        var = np.concatenate((np.full(np.shape(var), -a / 2), var), axis=1)
        newnodes = np.concatenate((newnodes, var), axis=0)

        var = np.arange(-a / 2, e - a / 2, length)
        var = var.reshape((len(var), 1))
        var1 = np.concatenate((var, np.full(np.shape(var), d / 2)), axis=1)
        var2 = np.concatenate((var, np.full(np.shape(var), -d / 2)), axis=1)
        newnodes = np.concatenate((newnodes, var1, var2), axis=0)

    if dim == 3:
        c = size[2]

        meshgrid = np.meshgrid(np.arange(-b / 2, b / 2, length),
                               np.arange(-c / 2, c / 2, length))
        newnodes = np.concatenate((np.full(np.size(meshgrid[0]),
                                           a / 2).\
                                   reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[0].reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
                                  axis=1)

        meshgrid = np.meshgrid(np.arange(-b / 2, -d / 2, length),
                               np.arange(-c / 2, c / 2, length))
        meshgrid = np.concatenate(
            (np.full(np.size(meshgrid[0]), -a / 2).\
             reshape(np.size(meshgrid[0]), 1),
             meshgrid[0].reshape(np.size(meshgrid[0]), 1),
             meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
                axis=1)
        newnodes = np.concatenate((newnodes, meshgrid), axis=0)

        meshgrid = np.meshgrid(np.arange(d / 2, b / 2, length),
                               np.arange(-c / 2, c / 2, length))
        meshgrid = np.concatenate(
            (np.full(np.size(meshgrid[0]), -a / 2).reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[0].reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
                                  axis=1)
        newnodes = np.concatenate((newnodes, meshgrid), axis=0)

        meshgrid = np.meshgrid(np.arange(-d / 2, d / 2, length),
                               np.arange(-c / 2, c / 2, length))
        meshgrid = np.concatenate(
            (np.full(np.size(meshgrid[0]), e - a/2).reshape(np.size(meshgrid[0]),
                                                              1),
                                   meshgrid[0].reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
                                                              axis=1)
        newnodes = np.concatenate((newnodes, meshgrid), axis=0)

        meshgrid = np.meshgrid(np.arange(-a / 2, e - a / 2, length),
                               np.arange(-c / 2, c / 2, length))
        meshgrid1 = np.concatenate(
            (meshgrid[0].reshape(np.size(meshgrid[0]), 1),
             np.full(np.size(meshgrid[0]), -d / 2).reshape(np.size(meshgrid[0]), 1),
             meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
            axis=1)
        meshgrid2 = np.concatenate((meshgrid[0].reshape(np.size(meshgrid[0]), 1),
                                    np.full(np.size(meshgrid[0]),
                                            d / 2).reshape(np.size(meshgrid[0]), 1),
                                   meshgrid[1].reshape(np.size(meshgrid[1]), 1)),
                                   axis=1)
        newnodes = np.concatenate((newnodes, meshgrid1, meshgrid2), axis=0)

    nodes = np.concatenate((nodes, newnodes), axis=0)
    nodes = np.unique(nodes, axis=0)
    return nodes


@njit(parallel=True)
def get_S(faces, nodes, length):
    S = np.zeros(len(faces), np.float64)
    
    for If in prange(len(faces)):
        f = faces[If]
        # f = f[np.isfinite(f)]
        cont = np.sum(np.sqrt(np.sum((nodes[f[1:]] - nodes[f[:-1]])**2,
                                     axis=1)))
        cont += np.sqrt(np.sum((nodes[f[0]] - nodes[f[-1]])**2))
        S[If] += cont / (1.3*length*.7 * len(f))
    return S


def node_mod(arg):
    import numpy as np
    
    def v_init(face, eo, nodes):  
        v = np.zeros((len(face), 3), np.float64)
        for i in prange(len(face)):
            e0 = face[i]
            e1 = np.concatenate((eo[eo[:,0]==e0,1], 
                                 eo[eo[:,1]==e0,0]))
            if len(e1) != 0: 
                v[i] = np.sum(nodes[e1,:], axis=0)/len(e1) - nodes[e0]
        return v
        
    def cont_init(face, nodes):
        ctr = np.sum(nodes[face], axis=0)/len(face)
        vecs = nodes[face] - ctr
        vecs /= np.stack((np.sqrt(np.sum(vecs**2, axis=1)),
                          np.sqrt(np.sum(vecs**2, axis=1)),
                          np.sqrt(np.sum(vecs**2, axis=1))), axis=1)
        
        cont = np.sum(np.sqrt(np.sum((nodes[face[1:]] - nodes[face[:-1]])**2, 
                                     axis=1)))
        cont += np.sqrt(np.sum((nodes[face[0]] - nodes[face[-1]])**2))
        return cont
    
    def incr_cont(n):
        cont = np.sum(np.sqrt(np.sum((n[1:] - n[:-1])**2,
                                     axis=1)))
        cont += np.sqrt(np.sum((n[0] - n[-1])**2))
        return cont
    
    def incr_nodes(nodes, v, length):
        n = nodes + v/length/10
        return n
    
    f, elems, nodes, length = arg
    elt = elems[np.sum(np.isin(elems, f), axis=1, dtype=bool)]
    eo = elt[np.prod(np.isin(elt, f), axis=1)!=1]
    
    v = v_init(f, eo, nodes)
    cont = cont_init(f, nodes)
    
    ii = 0 
    mod = v/length/10

    while ii<10 and cont < 1.33*length*.7 * len(f):
        n = incr_nodes(nodes[f], v, length)
        cont = incr_cont(n)
        mod = v/length/10        
        v = v_init(f, eo, nodes)
        l = np.sqrt(np.sum((n-np.roll(n, 1, axis=0))**2, axis=1))
        v[l>1.5*length] = 0
        v[np.roll(l, 1)>1.3*length] = 0
        ii += 1
        if (mod==0).all(): ii=10
        
    s = cont/(1.33*length*.7* len(f))
    return [s, mod]

# %% Lattice generation

def makepacking(dim: int, size: np.ndarray, length: float, des: float,
                prepacking: bool):
    """Generate random close packing in 2D or 3D.

    Parameters
    ----------
    dim: int
        Dimension of the packing
    size: np.ndarray
        Size of the domain
    length: float
        Average distance between beads
    des: float
        Disorder index. Keep <=0.2 to avoid bugs.
    prepacking: bool
        Use a pre-made packing and cut it to go faster, or create a new packing.

    Returns
    -------
    centers: np.ndarray
        Position of the bead centers in the packing.
        
    Examples
    --------
    Generation of a fast 2D packing
    
    >>> import treillis
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> pck = treillis.makepacking(2, np.array([30]), 5, .2, False, True)
    >>> # Observe the packing
    >>> plt.figure()
    >>> plt.plot(pck[:,0], pck[:,1], 'o')
    >>> plt.axis('equal')
    >>> plt.show()
    >>> # Observe the possible resulting lattice
    >>> plt.figure()
    >>> plt.triplot(pck[:,0], pck[:,1])
    >>> plt.axis('equal')
    >>> plt.show()
    
    """
    if dim==3:
        # packing construction:
        if des not in [0] and prepacking:
            prepacking=False
            print('No pre-packing with the wanted disorder index, switching to packing generation.')

        # box size:
        if len(size) == 1:
            side = (np.full((3), size[0]) * 2 + 2 * length) * (0.65 / 0.5)
            # 0.65/0.5 is the inverse of rescaling factor used in packing
            ## generation and which contracts the packing
        elif len(size) == 3:
            side = (size + 2 * length) * (0.65 / 0.5)
        else:
            side = (size[:3] + 2 * length) * (0.65 / 0.5)

        if prepacking:
            print('Using prepacking')
            nodes = np.loadtxt(
                os.path.abspath(
                    PREPACKPATH+'/Packing3D_nodes_' + str(float(des)) + r'.csv'),
                delimiter=',')[:, :3] * length

            # Extracting a random piece of the packing to still have different
            # lattices generated each time
            rng = np.random.default_rng()
            try:
                x = rng.uniform(min(nodes[:,0])+side[0],
                                max(nodes[:,0])-side[0], 1)
                y = rng.uniform(min(nodes[:,1])+side[1], 
                                max(nodes[:,1])-side[1], 1)
                z = rng.uniform(min(nodes[:,2])+side[2], 
                                max(nodes[:,2])-side[2], 1)
            except:
                x, y, z = nodes.mean(axis=0)
            
            nodes[:,0]-=x
            nodes[:,1]-=y
            nodes[:,2]-=z

            centers = nodes[
                (nodes[:,0]>=-side[0]/2) & (nodes[:,0]<side[0]/2)
                & (nodes[:,1]>=-side[1]/2) & (nodes[:,1]<side[1]/2)
                & (nodes[:,2]>=-side[2]/2) & (nodes[:,2]<side[2]/2)
                ]

        if not prepacking:
            print('Generating packing')
            path = PACKPATH
            test = os.listdir(path)
            for item in test:
                if item.endswith(".txt") or item.endswith(".xyzd") or\
                    item.endswith(".nfo") or item.endswith(".conf"):
                    os.remove(os.path.join(path, item))

            # log-normal distribution of radii
            Rmean = length/2
            Rstd = des * Rmean

            mu = np.log(Rmean ** 2 / np.sqrt(Rstd ** 2 + Rmean ** 2))
            sigma = np.sqrt(np.log(Rstd ** 2 / Rmean ** 2 + 1))

            # create an array of n diameters with a log-normal distribution
            
            # start with some already existing spheres to speed up
            Nd = np.prod(side) * .5 / (4/3 * np.pi * Rmean**3)
            Nd -= np.prod(side)/np.mean(side) / (4 * np.pi * Rmean**2)
            
            diameters = 2 * np.random.default_rng().lognormal(
                mu, sigma, size=int(Nd))
            vol = 4/3 * np.pi * np.sum((diameters/2)**3)
            while vol / np.prod(side) < 0.5:
                eq_r = np.cbrt(3/4 * vol*.5 / np.pi)
                Nd = 4 * np.pi * eq_r**2 / (4 * np.pi * Rmean**2)
                
                diameters = np.concatenate((diameters, 2 *\
                                            np.random.default_rng().lognormal(
                                                mu, sigma, size=int(Nd))))
                vol = np.sum((4 / 3) * np.pi * (diameters / 2) ** 3)

            N_particles = np.shape(diameters)[0]
            np.savetxt(path + r'\diameters.txt', diameters)

            launch_pack(path, N_particles, side, algo='fba', start=1)
            launch_pack(path, N_particles, side, algo='ls', start=0, erase=True)

            # when packing is generated, open the packing:
            packing_final = np.fromfile(path + r'\packing.xyzd')
            packing_final = packing_final.reshape(len(packing_final) // 4, 4)

            # Reading packing.nfo adn rescaling the packing:
            infos = pd.read_csv(path + r'\packing.nfo', sep='\t')
            TheoreticalPorosity = float(infos.iat[1, 0][23:])

            var = infos.iat[2, 0][16:]
            FinalPorosity = ''
            i = 0
            p = var[i]
            while p != ' ':
                FinalPorosity += p
                i += 1
                p = var[i]

            FinalPorosity = float(FinalPorosity)

            finalScalingFactor = ((1 - FinalPorosity)
                                  / (1 - TheoreticalPorosity)) ** (1 / 3)
            packing_final[:, :3] = packing_final[:, :3] / finalScalingFactor
            centers = packing_final[:, :3]
            centers = centers - 0.5 * side.reshape((1, 3))

    elif dim==2:
        if len(size) == 1:
            Radius = size[0] + 2 * length
        else:
            Radius = np.sqrt(2) * max(size[0], size[1]) / 2 + 2 * length

        # log-normal distribution of radiuses:
        Rmean = length
        Rstd = des * Rmean

        mu = np.log(Rmean ** 2 / np.sqrt(Rstd ** 2 + Rmean ** 2))
        sigma = np.sqrt(np.log(Rstd ** 2 / Rmean ** 2 + 1))

        if des not in [0, .1, .2, .3, .4, .5]:
            prepacking=False
            print('No pre-packing with the wanted disorder index, switching to packing generation.')

        if prepacking:
            try:
                print("Using prepacking")
                nodes = np.loadtxt(
                    os.path.abspath(
                        PREPACKPATH+'/Packing2D_nodes_' + str(float(des)) + r'.csv'),
                    delimiter=',')[:, :2] * length
                rng = np.random.default_rng()
                if Radius < np.sqrt(np.sum(np.ptp(nodes, axis=0)**2/4)):
                    x = rng.uniform(min(nodes[:,0])+Radius, 
                                    max(nodes[:,0])-Radius, 1)
                    y = rng.uniform(min(nodes[:,1])+Radius, 
                                    max(nodes[:,1])-Radius, 1)
                else: x, y = nodes.mean(axis=0)
                nodes[:,0]-=x
                nodes[:,1]-=y
    
                centers = nodes
            except:
                print('Prepacking failed, creating packing')
                prepacking=False

        if not prepacking:
            print('Generating packing')
            Rmean = length/2
            Rstd = des * Rmean

            mu = np.log(Rmean ** 2 / np.sqrt(Rstd ** 2 + Rmean ** 2))
            sigma = np.sqrt(np.log(Rstd ** 2 / Rmean ** 2 + 1))        
                
            centers = pack_2D(size, Radius, Rmean, mu, sigma, Rmean-Rstd, 
                    Rmean+2*Rstd)

    print('Packing generated')
    return centers


def frompacking(pack: np.ndarray, lattype: str, size: np.ndarray,
                length: float, dim: int, easy: bool = True):
    """
    Generate nodes and elems tables from a packing, for Voronoi and Delaunay lattices.

    Parameters
    ----------
    pack : np.ndarray
        Position of beads in RCP.
    lattype : str
        Type of the lattice.
    size : np.ndarray
        size of the domain.
    length : float
        Length of elements.
    dim: int, 2 or 3
        dimension of the lattice.

    Returns
    -------
    nodes_kept : TYPE
        Array of the position of nodes.
    elem_kept : TYPE
        Array of the elements.
    fixed : TYPE
        List of indices of fixed nodes.
        
        
    Examples
    --------
    Generation of a Delaunay-based lattice from a prepacking
    
    >>> import treillis
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Recover a pregenerated 1000x1000 packing
    >>> pck = treillis.makepacking(2, np.array([30]), 5, .2, True, False)
    >>> nodes, elems, fixed = treillis.frompacking(pck, 'TriDT3D', 
                                                np.array([30]), 5, 2)
    >>> elems = elems[:,:2].astype(int)
    >>> # Observe the resulting proto-lattice
    >>> plt.plot(nodes[elems, 0].T, nodes[elems, 1].T, 'k-')
    >>> plt.plot(nodes[fixed,0], nodes[fixed,1], 'ro') # boundaries
    >>> plt.axis('equal')
    >>> plt.show()
    

    """    
    if len(size) == 5:
        pack = addnodes2side(pack, size, 3, 2 * length)

    if lattype in ['Vor', 'Vor3D']:
        
        # creation of the Voronoi diagram:
        if dim == 3:
            # Creation of a Delaunay to parametrize the Voronoi better
            dt = Delaunay(pack)
            
            elem1 = dt.simplices.ravel().astype(int)
            elem2 = np.roll(dt.simplices, -1, axis=1).ravel().astype(int)
            minl = np.min(np.sqrt(np.sum(
                (pack[elem2] - pack[elem1])**2, axis=1)))
            
            if easy:
                options = 'Qx C-' + str(minl/100) + ' Q1 Q2 Q7 Q12 Q14 PF'\
                    + str(minl**2/2) + ' Qc Qz'
            else: options = 'Qx Q1 Q2 Q7 Q12 Q14 Qc Qz'
            vor = Voronoi(pack, qhull_options=options)
        else: 
           vor = Voronoi(pack)
        

        print("Extracting Voronoi cells")
        nodes_save = np.zeros((0, 3), dtype=TYPEVOR)
        elems_save = []#np.zeros((0, 2), dtype=TYPEVOR)

        for plane in vor.ridge_vertices:
            test = np.array(plane)

            elems = np.concatenate((test.reshape(len(test), 1),
                                    np.roll(test, -1).reshape(len(test), 1)),
                                   axis=1)
            # elems_save = np.concatenate((elems_save, elems), axis=0)
            elems_save.append(elems)
        elems_save = np.concatenate(elems_save, axis=0)

        nodes_save = vor.vertices
        elems_save = elems_save[
            (elems_save[:,0]!=-1) & (elems_save[:,1]!=-1)]
        elems_save.sort(axis=1)
        elems_save = np.unique(elems_save, axis=0)

        if dim == 3 and easy:
            print('Swelling the tiny faces')
            faces = [np.array(f) for f in vor.ridge_vertices if -1 not in f]
            faces = [f for f in faces if len(f)>2]

            niter = 0
            S = get_S(faces, nodes_save, length)
            t=0
            with ipp.Cluster(n=int(os.cpu_count()*2/3)) as rc:
                t0 = time()
                while niter < 10 and (np.array(S)<1).any() and t < 3*60:
                    print('... iteration #', niter+1)
                    niter += 1
    
                    S = []                    
                    dview = rc[:]

                    args = [[f, elems_save, nodes_save, length] for f in faces];
                    res = dview.map_async(node_mod, args);
                    res.wait_interactive()
                    
                    for i in range(len(res)):
                        r = res[i]
                        S.append(r[0])
                        nodes_save[faces[i]] += r[1]

                    t = time()-t0
                            
            del args, res
        print("\n---------------------------\nErasing supplementary nodes")
        # Erase the extraneous nodes generated by Voronoi and renumber the elems
        I_save = elems_save.reshape(2*len(elems_save)).astype(int)
        I_save = np.unique(I_save)
        I_save = np.sort(I_save)

        nodes_save = nodes_save[I_save]
        test0 = np.array([e0 in I_save for e0 in elems_save[:,0]])
        test1 = np.array([e1 in I_save for e1 in elems_save[:,1]])
        elems_save = elems_save[test0 & test1]

        _, e, i = np.intersect1d(elems_save[:,0],I_save, return_indices=True)
        elems_save[e, 0] = np.arange(len(I_save))[i]
        _, e, i = np.intersect1d(elems_save[:,1],I_save, return_indices=True)
        elems_save[e, 1] = np.arange(len(I_save))[i]

        newel = np.zeros((len(elems_save),2,dim))

        newel[:,0,:] = nodes_save[elems_save[:,0].astype(int),:]
        newel[:,1,:] = nodes_save[elems_save[:,1].astype(int),:]

        print('Deleting duplicated nodes')
        nodes_save = delete_dupl_nodes(nodes_save, elems_save, length)

        elems_save = newel

    elif lattype in ['TriDT', 'TriDT3D']:
        # creation of the Delaunay tesselation:
        nodes_save = pack
        DT = Delaunay(nodes_save)
        elem1 = DT.simplices.ravel()
        elem2 = np.roll(DT.simplices, -1, axis=1).ravel()
        elems_save = np.concatenate((elem1.reshape((len(elem1), 1)),
                                     elem2.reshape((len(elem2), 1))), axis=1)
        elems_save.sort(axis=1)
        elems_save = np.unique(elems_save, axis=0)

    # we find the elems inside the domain:
    elem_kept, elem_kept1, elem_kept2 = find_elems_inside(size, nodes_save,
                                                        elems_save)
    # we find the elems and then the nodes on the boundary:
    # try: 
        
    elm = elems_save
    if len(elm.shape)==2:
        elm = nodes_save[elems_save,:]
    fixed = find_nodes_bound(size, nodes_save, elm, 
                             elem_kept1.astype(int), elem_kept2.astype(int))
    
    # we remove the nodes outside the domain:
    nodes_kept = find_nodes_in(size, nodes_save)

    # we find the new indexes of elems according to the new nodes array:
    elem_kept, fixed = find_indices(nodes_kept, elem_kept, fixed, length)
    fixed = fixed.astype(int)
    return nodes_kept, elem_kept, fixed


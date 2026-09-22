"""
Fracture
========

Convenient functions to study the fracture of element networks.
"""
from copy import deepcopy
import numpy as np

__all__ = ["isbroken", "tensile_fracture", "compr_fracture", "equivalent_crack"]

#%% Function definition

def isbroken(network, bond):
    """
    Check if `bond` is the element leading to the network being fully broken in two pieces.
    
    Each of the parts must contain respectively the upper and lower fixed nodes.
    This ensures there is a real crack that cuts across the lattice.

    Parameters
    ----------
    network : treillis.mechanics.Element
        Element network to test.
    bond : int
        index of the element which is about to break

    Returns
    -------
    broken : bool
        True if the lattice is broken into 2 parts.
    """
    broken = False
    elems = deepcopy(network.elems[:,:2].astype(int))
    Nnodes = network.nodes.shape[0]
    elems = np.delete(elems, bond, axis=0)
    
    fixedup = network.fixed_nodes[network.nodes[network.fixed_nodes,1] 
                                  + network.L/5
                                  > max(network.nodes[network.fixed_nodes,1])]
    fixeddown = network.fixed_nodes[network.nodes[network.fixed_nodes,1] 
                                 - network.L/5
                                  < min(network.nodes[network.fixed_nodes,1])]
                                  

    for startingPoint in fixedup:
        visitedNodes = np.zeros((Nnodes), dtype=bool)

        nodeslinked = np.nonzero(np.isin(elems, startingPoint))
        visitedNodes[int(startingPoint)] = True
        nodesExamined = elems[nodeslinked[0],
                              (1 - nodeslinked[1]).astype(int)].astype(int)

        flag = True
        while flag:
            # check if nodes that we are currently looking at have already been visited
            if np.prod(visitedNodes[nodesExamined]):
                flag = False
            else:
                nodeslinked = np.nonzero(np.isin(elems,
                             nodesExamined[~visitedNodes[nodesExamined]]))
                visitedNodes[nodesExamined] = True
                nodesExamined = elems[nodeslinked[0],
                          (1 - nodeslinked[1]).astype(int)].astype(int)

        if np.sum(visitedNodes[fixeddown], dtype=bool):
            broken = False
            break

        broken = True
    return broken


def tensile_fracture(network, N: np.ndarray, M: np.ndarray = None,
                     V: np.ndarray = None,
                     weight: list = [1., 1., 1.]):
    """
    Compute the stress up to breaking stress ratio.

    Find the index of the element that is the next to break and the
    rescaling factor to adjust the boundary displacement at the very moment
    of breaking.
    
    Parameters
    ----------
    network : treillis.mechanics.elements.Element
        Network on which the fracture is calculated
    N : (Nelems,) numpy.ndarray
        Force to apply on the elements
    M : (Nelems,) numpy.ndarray
        Maximum torque for each element, needed for dof>1
    V : (Nelems,) or (Nelems,2) if dof>3 numpy.ndarray
        Shear forces, needed for dof>1
    weight : list, optional
        Weight coefficients for element breaking. Default are 1., 1., 1.

    Returns
    -------
    numBroken : INT
        index of the element that wil break.
    factor : FLOAT
        factor to rescale the boundary displacement so the element will be
        on the edge of breaking.
    """
    ymax = 0.5 * network.L / network.aspect_ratio
    W = network.inertia / ymax
    N = N.ravel()
    M = M.ravel()
    alphan, alpham, alphav = weight

    if network.dof == 1:
        arr = np.abs(N / network.failure_criterion)

    else:
        if network.dim==2:
            V1 = V
            arr = np.sqrt((alphan*N/network.section_area + alpham*M/W) ** 2\
                          + 3 * (alphav * V1 / network.section_area) ** 2)
        else:
            V1 = V[:,0]
            V2 = V[:,1]
            arr = np.sqrt((alphan* N/network.section_area + alpham*M/W)** 2\
                          + 3 * (alphav * V1 / network.section_area) ** 2\
                              + 3 * (alphav * V2 / network.section_area) ** 2)
        
        arr = arr / network.failure_criterion

    numBroken = np.argmax(arr, axis=0)
    factor = np.max(arr)
    return numBroken, factor    


def compr_fracture(network, E: float, sigma_prev: np.ndarray, 
                   sigma_prev_euler: np.ndarray, 
                   weight: list = [1., 1., 1.]):
    """
    Compute the stress up to breaking stress ratio in compression.

    Find the index of the element that is the next to break and the
    rescaling factor to adjust the boundary displacement at the very moment
    of breaking.

    Parameters
    ----------
    network : treillis.mechanics.elements.Element
        Network on which the fracture is calculated
    E : float
        Bulk Young's modulus
    sigma_prev : numpy.ndarray
        array that contains the stress state in the beams after previous beam 
        plastification
    sigma_prev_euler : numpy.ndarray
        array that contains the Euler stress state in the beams after previous
        beam plastification
    weight : list, optional
        Weight coefficients for element breaking. Default values are 1., 1., 1.

    Returns
    -------
    dict
        Results from the compression test:
        int
            index of the element that wil break.
        float
            factor to rescale the boundary displacement so the element will be on 
            the edge of breaking.
        numpy.ndarrays
            array of current (for fuse lattice) or stress (for spring and beam
            lattice) in the elements without the broken element.
        numpy.ndarray
            array of current (for fuse lattice) or stress (for spring and beam
            lattice) in the elements with the broken element.
        numpy.ndarray
            array of Euler stress (for spring and beam lattice) in the elements
            without the broken element.
        numpy.ndarray
            array of Euler stress (for spring and beam lattice) in the elements
            with the broken element.
        bool
            True if the bond fails under buckling, False if the failure is by
            yielding.
    """
    ymax = 0.5 * network.L / network.aspect_ratio
    
    W = network.inertia / (ymax)

    # calculation of the Euler critical stress
    sigma_euler = np.pi**2 * E * network.inertia\
        / (network.section_area * 0.25 * network.elems[:, 2]**2)

    alphan, alpham, alphav = weight

    if network.dof == 1:
        stressbefore = network.force
        arr = np.abs(stressbefore / (network.failure_criterion - sigma_prev))

    else:
        N = network.force
        M, V = network.shear_bend()
        if network.dim==2:
            V1 = V
            stressbefore = np.sqrt((alphan * N / network.section_area
                                    + alpham * M / W) ** 2 
                                   + 3 * (alphav*V1 / network.section_area)
                                   ** 2)
        else:
            V1 = V[:,0]
            V2 = V[:,1]
            stressbefore = np.sqrt((alphan * N / network.section_area 
                                    + alpham * M / W) ** 2 + 
                                   3 * (alphav * V1 / network.section_area) 
                                   ** 2 
                                   + 3 * (alphav * V2 / network.section_area)
                                   ** 2)

        stressbefore2 = N / network.section_area
        arr = stressbefore / (network.failure_criterion - sigma_prev)
        arr2 = stressbefore2 / (sigma_euler - sigma_prev_euler)

    if np.amax(arr) >= np.amax(arr2):
        numBroken = np.argmax(arr)
        factor = np.amax(arr)
        print('no buckling for beam of index ' + str(numBroken))
        buckling = False
    else:
        numBroken = np.argmax(arr2)
        factor = np.amax(arr2)
        buckling = True
        print('buckling for beam of index ' + str(numBroken))

    stressbefore *= 1/factor
    stressafter = np.delete(stressbefore, numBroken)
    stressbefore2 *= 1/factor
    stressafter2 = np.delete(stressbefore2, numBroken)

    res = {'index_broken': numBroken,
           'factor': factor,
           'stress': stressafter,
           'init_stress': stressbefore,
           'euler': stressafter2,
           'init_euler': stressbefore2,
           'buckling':buckling}
    return res


def equivalent_crack(E: float, force: float, displ: float, nw_size: np.ndarray):
    """
    Calculate the equivalent crack length if the sample was homogeneous.

    Parameters
    ----------
    E : float
        Young's modulus of the metamaterial.
    force : float
        External force.
    displ : float
        Displacement.
    nw_size : numpy.ndarray
        3D-size of the sample

    Returns
    -------
    float
        Equivalent crack length for a straight crack in a homogeneous sample.
    
    Note
    ----
    The extarnal force is the sum of the force value for the boundary nodes on 
    one side, measured at boundary application.

    """
    geom = ''
    if len(nw_size) > 1:
        if nw_size[0] >= 2*nw_size[1]:
            if len(nw_size) == 2:
                geom = 'thin strip'
            else:
                if nw_size[2] <= nw_size[0]/10:
                    geom = 'thin strip'
    if geom == 'thin strip' and len(nw_size)==3:
        k = force / (2 * displ)
        return nw_size[0] - nw_size[1] * k / (nw_size[2] * E)
    else: 
        print("Sample shape hasn't been implemented yet.")
        return 0

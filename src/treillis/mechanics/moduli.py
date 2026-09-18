# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:17:43 2026

@author: EG283299
"""

"""
Moduli
======

Calculation of mechanical moduli of lattices.
"""

#%% Importations
from copy import deepcopy
import numpy as np
import ipyparallel as ipp

from scipy.spatial import ConvexHull, Voronoi
from scipy.optimize import minimize

from .elements import Element

TYPEFLOAT = np.float64

__all__ = ['compliance']

#%%

def comp_solver(dim, n, sigma, epsilon):
    """
    Compute the compliance tensor in Voigt notation.
    
    To obtain the tensor, we solve the tensor equation
    epsilon = S:sigma.
    For the 2D case, plane stress is assumed. 
    In 2D: [1/Ex      -NUyx/Ey   ##
            -NUxy/Ex   1/Ey      ##
            ##          ##       1/Gxy] 
    
    In 3D: [1/Ex      -NUyx/Ey  -NUzx/Ez  ##     ##     ##
            -NUxy/Ex  1/Ey      -NUzy/Ez  ##     ##     ##
            -NUxz/Ex  -NUyz/Ey  1/Ez      ##     ##     ##
            ##        ##        ##        1/Gyz
            ##        ##        ##        ##     1/Gzx  ##
            ##        ##        ##        ##     ##     1/Gxy]
        
        
        Where ## are cross-terms with less interest. 

    Parameters
    ----------
    dim : int
        Dimension of the problem
    n : int
        number of Voronoi cells
    sigma : (Nvor, Nexp * Ndim, Nexp * Ndim) np.ndarray
        concatenation of stress tensors calculated over Voronoi cells on 
        the non-boundary nodes.
    epsilon : (Nvor, Nexp * Ndim, Nexp * Ndim) np.ndarray
        concatenation of stress tensors calculated over Voronoi cells on 
        the non-boundary nodes.
        
    Returns
    -------
    compliance : np.ndarray
        Compliance tensor.

    """
    A = np.zeros((6 + dim//3 * 15, 6 + dim//3 * 15))
    B = np.zeros((6 + dim//3 * 15))   
    B[:(3*(dim-1))] = np.sum(sigma * epsilon, axis=0)
    
    final_compliance = np.zeros((3*(dim-1), 3*(dim-1)), dtype=TYPEFLOAT)
    
    if dim == 2:
            A[0,:3] = np.sum(sigma * sigma[:,0].reshape(n, 1), axis=0)
            A[1, np.array([1,3,4])] = np.sum(sigma * sigma[:,1].reshape(n,1),
                                             axis=0)
            A[2, np.array([2,4,5])] = np.sum(sigma * sigma[:,2].reshape(n,1),
                                             axis=0)
            A[3, np.array([1,3,4])] = np.sum(sigma * sigma[:,0].reshape(n,1), 
                                             axis=0)
            A[3, np.array([0,1,2])] += np.sum(sigma * sigma[:,1].reshape(n,1), 
                                              axis=0)
            A[4, np.array([2,4,5])] = np.sum(sigma * sigma[:,1].reshape(n,1), 
                                             axis=0)
            A[4, np.array([1,3,4])] += np.sum(sigma * sigma[:,2].reshape(n,1),
                                              axis=0)
            A[5, np.array([2,4,5])] = np.sum(sigma * sigma[:,0].reshape(n,1), 
                                             axis=0)
            A[5, np.array([0,1,2])] += np.sum(sigma * sigma[:,2].reshape(n,1), 
                                              axis=0)  
            B[3:6] = np.sum(sigma * np.roll(epsilon, -1, axis=1)
                            + np.roll(sigma, -1, axis=1) * epsilon, axis=0)   
        
    else:
        A[0,:6] = np.sum(sigma * sigma[:,0].reshape(n,1), axis=0)
        A[1, np.array([1,6,7,8,9,10])] = np.sum(sigma*sigma[:,1].reshape(n,1), axis=0)
        A[2, np.array([2,7,11,12,13,14])] = np.sum(sigma * sigma[:,2].reshape(n,1), 
                                         axis=0)
        A[3, np.array([3,8,12,15,16,17])] = np.sum(sigma * sigma[:,3].reshape(n,1), 
                                         axis=0)
        A[4, np.array([4,9,13,16,18,19])] = np.sum(sigma * sigma[:,4].reshape(n,1), 
                                         axis=0)
        A[5, np.array([5,10,14,17,19,20])] = np.sum(sigma * sigma[:,5].reshape(n,1), 
                                          axis=0)   
        A[6,:6] = np.sum(sigma * sigma[:,1].reshape(n,1), axis=0)
        A[6, np.array([1,6,7,8,9,10])] += np.sum(sigma * sigma[:,0].reshape(n,1), axis=0)            
        A[7,:6] = np.sum(sigma * sigma[:,2].reshape(n,1), axis=0)
        A[7, np.array([2,7,11,12,13,14])] += np.sum(sigma * sigma[:,0].reshape(n,1),
                                          axis=0)
        A[8,:6] = np.sum(sigma * sigma[:,3].reshape(n,1), axis=0)
        A[8, np.array([3,8,12,15,16,17])] += np.sum(sigma * sigma[:,0].reshape(n,1), 
                                          axis=0)
        A[9,:6] = np.sum(sigma * sigma[:,4].reshape(n,1), axis=0)
        A[9, np.array([4,9,13,16,18,19])] += np.sum(sigma * sigma[:,0].reshape(n,1), 
                                          axis=0)
        A[10,:6] = np.sum(sigma * sigma[:,5].reshape(n,1), axis=0)
        A[10, np.array([5,10,14,17,19,20])] += np.sum(sigma * sigma[:,0].reshape(n,1), 
                                            axis=0)            
        A[11, np.array([1,6,7,8,9,10])] = np.sum(sigma * sigma[:,2].reshape(n,1), axis=0)
        A[11, np.array([2,7,11,12,13,14])] += np.sum(sigma * sigma[:,1].reshape(n,1),
                                           axis=0)    
        A[12, np.array([1,6,7,8,9,10])] = np.sum(sigma * sigma[:,3].reshape(n,1), axis=0)
        A[12, np.array([3,8,12,15,16,17])] += np.sum(sigma * sigma[:,1].reshape(n,1),
                                           axis=0)
        A[13, np.array([1,6,7,8,9,10])] = np.sum(sigma * sigma[:,4].reshape(n,1), axis=0)
        A[13, np.array([4,9,13,16,18,19])] += np.sum(sigma * sigma[:,1].reshape(n,1), 
                                           axis=0)
        A[14, np.array([1,6,7,8,9,10])] = np.sum(sigma * sigma[:,5].reshape(n,1), axis=0)
        A[14, np.array([5,10,14,17,19,20])] += np.sum(sigma * sigma[:,1].reshape(n,1), 
                                            axis=0)            
        A[15, np.array([2,7,11,12,13,14])] = np.sum(sigma * sigma[:,3].reshape(n,1), 
                                          axis=0)
        A[15, np.array([3,8,12,15,16,17])] += np.sum(sigma * sigma[:,2].reshape(n,1),
                                           axis=0)
        A[16, np.array([2,7,11,12,13,14])] = np.sum(sigma * sigma[:,4].reshape(n,1),
                                          axis=0)
        A[16, np.array([4,9,13,16,18,19])] += np.sum(sigma * sigma[:,2].reshape(n,1), 
                                           axis=0)
        A[17, np.array([2,7,11,12,13,14])] = np.sum(sigma * sigma[:,5].reshape(n,1),
                                          axis=0)
        A[17, np.array([5,10,14,17,19,20])] += np.sum(sigma * sigma[:,2].reshape(n,1),
                                            axis=0)            
        A[18, np.array([3,8,12,15,16,17])] = np.sum(sigma * sigma[:,4].reshape(n,1),
                                          axis=0)
        A[18, np.array([4,9,13,16,18,19])] += np.sum(sigma * sigma[:,3].reshape(n,1),
                                           axis=0)
        A[19, np.array([3,8,12,15,16,17])] = np.sum(sigma * sigma[:,5].reshape(n,1), 
                                          axis=0)
        A[19, np.array([5,10,14,17,19,20])] += np.sum(sigma * sigma[:,3].reshape(n,1),
                                            axis=0)            
        A[20, np.array([4,9,13,16,18,19])] = np.sum(sigma * sigma[:,5].reshape(n,1), axis=0)
        A[20, np.array([5,10,14,17,19,20])] += np.sum(sigma * sigma[:,4].reshape(n,1),
                                            axis=0)
        
        B[6:11] = np.sum(sigma[:,0].reshape(n,1) * epsilon[:,1:] + sigma[:,1:]
                         * epsilon[:,0].reshape(n,1), axis=0)
        B[11:15] = np.sum(sigma[:,1].reshape(n,1) * epsilon[:,2:] 
                          + sigma[:,2:] * epsilon[:,1].reshape(n,1), axis=0)
        B[15:18] = np.sum(sigma[:,2].reshape(n,1) * epsilon[:,3:] + sigma[:,3:] 
                          * epsilon[:,2].reshape(n,1), axis=0)
        B[18:20] = np.sum(sigma[:,3].reshape(n,1) * epsilon[:,4:] + sigma[:,4:] 
                          * epsilon[:,3].reshape(n,1), axis=0)
        B[20] = np.sum(sigma[:,4] * epsilon[:,5] + sigma[:,5] * epsilon[:,4],
                       axis=0)
        
    S = np.linalg.solve(A, B)

    final_compliance[np.triu_indices(3 * (dim-1))] = S
    final_compliance = final_compliance + final_compliance.T
    

    final_compliance[np.diag_indices(3 * (dim-1))] = final_compliance[
        np.diag_indices(3 * (dim-1))] / 2        

    return final_compliance


def compliance(global_stress: list[np.ndarray], 
               global_strain: list[np.ndarray], 
               local_stress: list[np.ndarray], 
               local_strain: list[np.ndarray]):
    """
    Compute the compliance tensor.
    
    The computation of compliances is achieved through 'best-fit' algorithm
    run over a set of mechanical tests (tractions and shearings). For 
    computations to complete, the system must be over-constrained, meaning
    Ndim +1 tests must be performed.

    Parameters
    ----------
    global_stress : list of np.ndarray
        global stress tensors for the different tests.
    global_strain : list of np.ndarray
        global strain tensors for the different tests.
    local_stress : list of np.ndarray
        local stress tensors for the different tests.
    local_strain : list of np.ndarray
        local strain tensors for the different tests.
        
    Raises
    ------
    ValueError
        a minimum number of independant tensors must be given in order
        to calculate all the elastic constants.

    Returns
    -------
    global_compliance : np.ndarray
        global compliance matrix in Voigt notation.
    local_compliance : ARRAY OF FLOAT
        local compliance matrix in Voigt notation.

    """  
    if np.shape(global_stress[0])[0] == 2:
        if len(global_stress) < 2:
            raise ValueError('Number of equations to solve the problem is not enough, you need at least the stress and strain tensors from 2 independant mechanical tests.')
    elif np.shape(global_stress[0])[0] == 3:
        if len(global_stress) < 4:
            raise ValueError('Number of equations to solve the problem is not enough, you need at least the stress and strain tensors from 4 independant mechanical tests.')
    else:
        raise ValueError('global_stress and global_strain are not properly passed. They must be passed as a list of tensors from different tests.')        
    
    def prepare(global_stress, global_strain, dim):
        sig = np.array([global_stress[::dim, 0], global_stress[1::dim, 1], 
                        0.5 * (global_stress[1::dim, 0]
                               + global_stress[::dim, 1])]).T
        eps = np.array([global_strain[::dim, 0], global_strain[1::dim, 1], 
                        global_strain[1::dim, 0] + global_strain[::dim, 1]]).T   
        
        if dim == 3:
            sig = np.array([global_stress[::dim, 0], global_stress[1::dim, 1], 
                            global_stress[2::dim, 2], 
                            0.5 * (global_stress[2::dim, 1] 
                                   + global_stress[1::dim, 2]), 
                            0.5 * (global_stress[2::dim, 0] 
                                   + global_stress[::dim, 2]), 
                            0.5 * (global_stress[1::dim, 0] 
                                   + global_stress[::dim, 1])]).T
            eps = np.array([global_strain[::dim, 0], global_strain[1::dim, 1],
                            global_strain[2::dim, 2], 
                            global_strain[2::dim, 1] + global_strain[1::dim, 2],
                            global_strain[2::dim, 0] + global_strain[::dim, 2], 
                            global_strain[1::dim, 0] + global_strain[::dim, 1]]).T
        return sig, eps
    
    dim = np.shape(global_stress[0])[0]
    n = len(global_stress)
    
    # Creation of a system of equations to solve for the stresses in the
    # global representation
    global_stress = np.concatenate(deepcopy(global_stress))
    global_strain = np.concatenate(deepcopy(global_strain))
    
    sig, eps = prepare(global_stress, global_strain, dim)
    
    global_compliance = comp_solver(dim, n, sig, eps)   

    # The same in the local representation
    def loc_compl(local_stress, dim, local_strain):
        local_compliance = np.zeros((np.shape(local_stress[0])[0], 
                                     3 * (dim - 1), 
                                     3 * (dim - 1)))
        
        for i in range(np.shape(local_stress[0])[0]):
            global_stress = np.concatenate([local_stress[k][i, :, :] 
                                            for k in range(len(local_stress))])
            global_strain = np.concatenate([local_strain[k][i, :, :] 
                                            for k in range(len(local_stress))])
    
            sig, eps = prepare(global_stress, global_strain, dim)
            
            local_compliance[i, :, :] = comp_solver(dim, n, sig, eps)
        return local_compliance
    
    try:
        local_compliance = loc_compl(local_stress, dim, local_strain)
        return global_compliance, local_compliance
    except: 
        print('Warning: only global compliance was computed')
        return global_compliance, []
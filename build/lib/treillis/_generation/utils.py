"""
Utilitaries
===========


Useful functions that don't really fit anywhere.
"""

#%% Import

import numpy as np

__all__ = ['vec_dist']

#%% Functions

def vec_dist(X, Y):
    """
    Calculate the Euclidian distance matrix between point with vector manipulation.
    
    This function can create very large tables that overload the memory. Even
    if it is much faster than looping over a table, it can be inefficient
    in that it can overload your memory.
    
    Parameters
    ----------
    X, Y : (NXval, Ndim), (NYval, Ndim) np.ndarray
        tables whose distance matrix you want to compute
    
    Returns
    -------
    (Npoints, Npoints) np.ndarray
        Distance matrix between all points.
        
    """
    X2 = np.sum(X**2, axis=1)
    Y2 = np.sum(Y**2, axis=1)
    
    XY = np.matmul(X, Y.T)
    
    dist = np.sqrt(X2[:,None].T + Y2[:,None] -2*XY)

    return dist
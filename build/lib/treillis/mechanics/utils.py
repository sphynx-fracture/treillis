"""
Mechanical utilitary functions
==============================

Repository of various useful functions for the mechanics module.
"""

import numpy as np

__all__ = ["inertia"]


def inertia(shape: np.ndarray, scale: float):
    """
    Calculate the bending inertia of an arbitrary cross-section

    Parameters
    ----------
    shape : np.ndarray
        (N, N) array of 1s and 0s where the 1s correspond to section of the element, and the 0 are empty.
    scale : float
        m/pixel scale.

    Returns
    -------
    inertia_min : float
        Minimal inertia of the section.
    inertia_max : float
        Maximal inertia of the section.
    inertia_tens : float
        Full inertia tensor.

    """
    x, y = np.meshgrid(np.linspace(-scale/2,scale/2,shape.shape[0]),
                         np.linspace(-scale/2,scale/2,shape.shape[1]))
    y*=-1
    
    xy = np.concatenate((
        x.reshape(x.shape[0], x.shape[1], 1),
        y.reshape(y.shape[0], y.shape[1], 1)), axis=2)
    
    inertia_tens = np.nan * np.ones((2, 2))
    centroid = np.array([np.mean(xy[:,:,0]*shape), np.mean(xy[:,:,1]*shape)])
    
    coords = np.concatenate((
        xy[:,:,0][shape.astype(bool)].reshape(np.sum(shape), 1),
        xy[:,:,1][shape.astype(bool)].reshape(np.sum(shape), 1)),
        axis=1)
    
    inertia_tens[0, 0] = np.sum((coords[:,0] - centroid[0]) ** 2)
    inertia_tens[1, 1] = np.sum((coords[:,1] - centroid[1]) ** 2)
    inertia_tens[0, 1] = np.sum((coords[:,0] - centroid[0]) 
                                * (coords[:,1] - centroid[1]))
    inertia_tens[1, 0] = inertia_tens[0, 1] 
    D, _ = np.linalg.eig(inertia_tens) 
    inertia_min = min(D)
    inertia_max = max(D)
    return inertia_min, inertia_max, inertia_tens


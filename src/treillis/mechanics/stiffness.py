"""
Stiffness
=========

Repository for local 2D and 3D stiffness tensors of implemented element physics.


The different elements can be:
    
- fuse: 
- spring
- beams

`phi` is the Timoshenko parameter used to modify standard stiffness matrixes in
order to use Timoshenko beam model rather than Euler-Bernouilli.
"""

#%% Importations
import numpy as np
cpufloat = np.float64
#%% Fuse
def fuse(device='cpu'):
    """Define the local stiffness matrix for the fuse element."""
    return np.array([[1, -1], [-1, 1]], dtype=cpufloat)
    

#%% Spring
def spring_1(dim, device='cpu'):
    """
    Define part of the stiffness matrix for a spring element.

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the spring element.

    """
    if dim == 2:
        mat = np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0],
                        [0, 0, 0, 0]], dtype=cpufloat)
    else:
        mat = np.zeros((6, 6), dtype=cpufloat)
        mat[0, 0] = 1
        mat[0, 3] = -1
        mat[3, 0] = -1
        mat[3, 3] = 1
            
    return mat


def spring_2(dim, phi, device='cpu'):
    """
    Define part of the stiffness matrix for a spring element.

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).
    phi : FLOAT
        Timoshenko shear coefficient.

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the spring element.

    """
    if dim == 2:
        mat = np.array([[0, 0, 0, 0], [0, 12, 0, -12], [0, 0, 0, 0],
                        [0, -12, 0, 12]], dtype=cpufloat)

    else:
        mat = np.zeros((6, 6), dtype=cpufloat)
        mat[1, 1] = 12
        mat[2, 2] = 12
        mat[4, 4] = 12
        mat[5, 5] = 12
        mat[1, 4] = -12
        mat[4, 1] = -12
        mat[2, 5] = -12
        mat[5, 2] = -12

    mat *= 1 / (1 + phi)
        
    return mat

#%% Beam
def beam_1(dim, device='cpu'):
    """
    Define the traction part of the stiffness matrix for a beam element.

    stiffness = E_S/l

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the beam element.

    """
    if dim == 2:
        return np.array([[1, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [-1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                        dtype=cpufloat)
    mat = np.zeros((12, 12), dtype=cpufloat)
    mat[0, 0] = 1
    mat[0, 6] = -1
    mat[6, 0] = -1
    mat[6, 6] = 1  
    
    return mat


def beam_2(dim, phi, device='cpu'):
    """
    Define part the flexion modulus matrix part for a beam element.

    stiffness = E_S*I/l

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).
    phi : FLOAT
        Timoshenko shear coefficient.

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the beam element.

    """
    if dim == 2:
        mat = np.array([[0, 0, 0, 0, 0, 0], [0, 12, 0, 0, -12, 0],
                        [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, -12, 0, 0, 12, 0], [0, 0, 0, 0, 0, 0]],
                       dtype=cpufloat)

    else:
        mat = np.zeros((12, 12), dtype=cpufloat)
        mat[1, 1] = 12
        mat[2, 2] = 12
        mat[7, 7] = 12
        mat[8, 8] = 12
        mat[1, 7] = -12
        mat[7, 1] = -12
        mat[2, 8] = -12
        mat[8, 2] = -12

    mat *= 1 / (1 + phi)

    return mat


def beam_3(dim, phi, device='cpu'):
    """
    Define the torsion along the first axis part of the stiffness matrix for a beam element.

    stiffness = E_S*I/l along first axis

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).
    phi : FLOAT
        Timoshenko shear coefficient.

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the beam element.

    """
    if dim == 2:
        mat = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 6, 0, 0, 6],
                        [0, 6, 0, 0, -6, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, -6, 0, 0, -6], [0, 6, 0, 0, -6, 0]],
                       dtype=cpufloat)

    else:
        mat = np.zeros((12, 12), dtype=cpufloat)
        mat[4, 2] = -6
        mat[2, 4] = -6
        mat[5, 1] = 6
        mat[1, 5] = 6
        mat[7, 5] = -6
        mat[5, 7] = -6
        mat[8, 4] = 6
        mat[4, 8] = 6
        mat[10, 2] = -6
        mat[2, 10] = -6
        mat[11, 1] = 6
        mat[1, 11] = 6
        mat[10, 8] = 6
        mat[8, 10] = 6
        mat[11, 7] = -6
        mat[7, 11] = -6

    mat *= 1 / (1 + phi)
        
    return mat
    

def beam_4(dim, phi, device='cpu'):
    """
    Define the torsion along second axis part of the stiffness matrix for a beam element.

    Parameters
    ----------
    dim : INT
        dimension of the lattice (2 for 2D and 3 for 3D).
    phi : FLOAT
        Timoshenko shear coefficient.

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the beam element.

    """
    if dim == 2:
        return np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, (4 + phi) / (1 + phi), 0, 0,
                          (2 - phi) / (1 + phi)], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, (2 - phi) / (1 + phi), 0,
                                              0, (4 + phi) / (1 + phi)]],
                        dtype=cpufloat)

    mat = np.zeros((12, 12), dtype=cpufloat)
    mat[4, 4] = (4 + phi) / (1 + phi)
    mat[5, 5] = (4 + phi) / (1 + phi)
    mat[10, 10] = (4 + phi) / (1 + phi)
    mat[11, 11] = (4 + phi) / (1 + phi)
    mat[4, 10] = (2 - phi) / (1 + phi)
    mat[10, 4] = (2 - phi) / (1 + phi)
    mat[11, 5] = (2 - phi) / (1 + phi)
    mat[5, 11] = (2 - phi) / (1 + phi)
    
    return mat


def beam_5(device='cpu'):
    """
    Define the shear part of the stiffness matrix for a beam element.

    stiffness = G*J/l

    Returns
    -------
    mat : ARRAY OF FLOAT
        Part of the local stiffness matrix for the beam element in 3D.

    """
    mat = np.zeros((12, 12), dtype=cpufloat)
    mat[3, 3] = 1
    mat[3, 9] = -1
    mat[9, 3] = -1
    mat[9, 9] = 1
    
    return mat
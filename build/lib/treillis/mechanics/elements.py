"""
Elements
========

Creation of tables with the elastic behavior of all elements.

The memory of the network is saved in a h5 file, using pytables. Future 
versions will allow calculatons either on CPU or GPU, once I figure out an
efficient read-compute-write pipeline.
"""
import os
from copy import deepcopy
# from time import time
# from functools import property

import numpy as np
# import cupy as cp

import tables as tb
from numba import jit, prange
from scipy import sparse
from scipy.spatial import Voronoi, ConvexHull
from sksparse.cholmod import cholesky
import scipy.sparse.linalg as lng
import scipy.linalg.interpolative as sli
# from cholespy import CholeskySolverD, MatrixType

from . import stiffness
from .utils import inertia
import treillis

__all__ = ["Element", "Fuse", "Spring", "Beam", "shear_bend"]

#%% Base element class  

class Element:
    """
    Representation of linear mechanical experiments over the lattices.
    
    Based on the geometrical informations of the elements in the lattice, 
    global and local stiffness are computed, and using various matrix inversion
    methods, the elastic energy can be recovered. Elements keep in memory the
    previous deformation applied.
    
    Parameters
    ----------
    lattice : treillis.Lattice
        Lattice the element network is built from
    dof : int
        Number of degrees of freedom of the network    
    location : str, optional
        Folder where the memory will be saved. Default is the working folder.
    
    Attributes
    ----------
    alpha : (Nelems,1) numpy.ndarray
        Corrective term in case of Timoshenko beams, otherwise 0-array
    angle : (Nelems, Ndim-1) np.ndarray
        Angle of the elements, polar angle if 2D, or 3D angles.
    ang_stiffness : (Nelems,1) numpy.ndarray
        Angular stiffness of the elements (0-array if dof<3)
    aspect_ratio : (Nelems,1) numpy.ndarray
        Aspect ratio of the elements gotten from the lattice
    ax_stiffness : (Nelems,1) numpy.ndarray
        Axial stiffness of the elements
    beam_section : numpy.ndarray or str
        Section of the elements gotten from the lattice
    boundaries : (N boundary nodes, dof, 2) numpy.ndarray
        Array containing the boundary nodes and their imposed displacement. Boundary nodes are anchored for an inf displacement.
    crack : (N cracked element, 2) numpy.ndarray
        Array containing the node indices around the crack.
    del_elm : numpy.ndarray
        Array containing the indices of the deleted elements with th ecrack progression.
    dim : int
        Dimensionality of the lattice
    displacement : (N nodes, dof) numpy.ndarray
        Nodes displacement. If dof>3, includes also the rotations.
    dof : int
        Degrees of freedom of the system
    elems : numpy.ndarray
        Elements as defined by the nodes indices and the length.
    failure_criterion : (Nelems,1) numpy.ndarray
        (N elems, 1) array containing the values to check whether or not an element is broken.
        TODO: find some way to implement it meaningfully.
    fixed_nodes : (N fixed,) numpy.ndarray
        Array containing the indices of the boundary nodes of the original lattice.
    K_global : (N nodes, N nodes) scipy.sparse.lil_array
        Global stiffness of the network. 
    L : float
        Average length of the lattice elements
    length : (Nelems,1) np.ndarray
        Length of each element.
    mempath : str
        Location of the h5 memory file of the test. 'off' to avoid saving
        the memory, for example in case of parallel computing.
    nodes : (N nodes, N nodes) numpy.ndarray
        Position of the network node.
    Ntest : int
        Number of tests that have been performed.
    section_area : (Nelems,1) numpy.ndarray
        Section area of the lattice elements.
    force : (Nelems,1) numpy.ndarray
        array containing the line tension in the network elements
    tors_stiffness : (Nelems,1) numpy.ndarray
        array containing the torsional stiffness of the elements. 0-array for dof<=3
    type : str
        Type of Element network
    
    See Also
    --------
    treillis.mechanics.elements.Fuse
    treillis.mechanics.elements.Spring
    treillis.mechanics.elements.Beam    
    """
    
    def __init__(self, lattice: treillis.Lattice, dof: int, location: str = None):
        self.dim = lattice.dim
        self.sizedom = lattice.sizedom
        self.nodes = deepcopy(lattice.nodes)
        self.elems = deepcopy(lattice.elems)
        self.force = np.zeros(self.elems.shape[0])
        self.ax_stiffness = np.zeros(self.elems.shape[0])
        self.ang_stiffness = np.zeros(self.elems.shape[0])
        self.failure_criterion = np.zeros(self.elems.shape[0])
        self.tors_stiffness = np.zeros(self.elems.shape[0])
        self.alpha = np.zeros(self.elems.shape[0])
        self.inertia = np.zeros(self.elems.shape[0])

        self.fixed_nodes = np.array(lattice.fixed_nodes)

        self.L = lattice.L
        self.aspect_ratio = lattice.aspect_ratio

        self.K_global = sparse.lil_array(
            (dof * self.nodes.shape[0], 
             dof * self.nodes.shape[0]),
            dtype=float)
        self.dof = dof
        self.boundaries = np.zeros((2, self.dof*self.nodes.shape[0]))
        self.displacement = np.zeros((self.nodes.shape[0], self.dof))
        self.type = 'Element'
        self.beam_section = lattice.beam_section
        
        self.crack = np.ones((1, 2)) * np.nan
        
        print(location)
        if location != 'off':
            if location is None: location = os.getcwd()
            self.mempath = os.path.abspath(location + r'/' + 'memory.h5')
            print('Memory will be saved at', self.mempath)
            memory = tb.open_file(self.mempath, 'w', driver='H5FD_CORE')
            
            group = memory.create_group('/', 'init')
            memory.create_carray(group, 'elements',
                                      obj=self.elems[:,:2].astype(int))
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            memory.create_carray(group, 'angular_stiffness', 
                                      obj=self.ang_stiffness)
            memory.create_carray(group, 'torsional_stiffness', 
                                      obj=self.tors_stiffness)
            memory.create_carray(group, 'Timoshenko', 
                                      obj=self.alpha)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)
            memory.create_carray(group, 'aspect_ratio', 
                                      obj=self.aspect_ratio)
            if isinstance(self.beam_section, np.ndarray):
                memory.create_carray(group, 'beam_section', 
                                          obj=self.beam_section)
            memory.create_carray(group, 'inertia', 
                                      obj=self.inertia)
            memory.close()
        else: self.mempath = 'off'
       
        self.last_broken = {}
        self.Ntest = 1
        self.del_elm = np.array([])
        
    @property
    def section_area(self):
        if self.beam_section == 'Circular':
            section_area = np.pi * (self.L/self.aspect_ratio/2)**2
        if self.beam_section == 'Square':
            section_area = (self.L/self.aspect_ratio)**2
        if isinstance(self.beam_section, np.ndarray):
            section_area = np.sum(self.beam_section, axis=(0, 1))\
                * (self.L/self.aspect_ratio / self.beam_section.shape[0])**2  
        return section_area
        
        
    @property
    def length(self):
        L = np.sqrt(np.sum(
            (self.nodes[self.elems[:,1].astype(int)]
             - self.nodes[self.elems[:,0].astype(int)])**2, axis=1
            ))
        return L
        
    @property
    def angle(self):
        dd = self.nodes[self.elems[:,1].astype(int)]\
            - self.nodes[self.elems[:,0].astype(int)]
        angle = np.arctan2(dd[:,1], dd[:,0]).reshape(self.elems.shape[0])
        if self.dim == 3:
            angle = np.concatenate((
                angle.reshape(len(angle), 1), 
                np.arctan2(np.sqrt(np.sum(dd[:,:2]**2, axis=1)),
                           dd[:,2]).reshape(self.elems.shape[0],1)),
                axis=1)
        return angle.reshape(self.elems.shape[0], self.dim-1)
    
    def update_memory(self):
        if self.mempath != 'off':
            memory = tb.open_file(self.mempath, 'a')
            group = memory.create_group('/', 'iteration'+str(self.Ntest))
            if not 0 in self.del_elm.shape:
                memory.create_carray(group, 'deleted_elements',
                                          obj=self.del_elm)
            memory.create_carray(group, 'displacement',
                                      obj=self.displacement)
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)       
            if self.dim > 1:
                memory.create_carray(group, 'angular_stiffness', 
                                          obj=self.ang_stiffness)        
            if self.dim > 3:
                memory.create_carray(group, 'torsional_stiffness', 
                                          obj=self.tors_stiffness)            
            if (self.alpha!=0).any():    
                memory.create_carray(group, 'Timoshenko', 
                                          obj=self.alpha)        
            memory.close()
        
        pass
    
    def reinitialize(self):
        """Erase the entire history of experiments and go back to initial state."""
        if self.mempath != 'off':
            print('Reinitializing')
            self.Ntest = 1
            memory = tb.open_file(self.mempath, 'r')
            self.elems = memory.root.init.elements[:]
            self.force = memory.root.init.force[:]
            self.failure_criterion = memory.root.init.failure_criterion[:]
            self.ax_stiffness = memory.root.init.axial_stiffness[:]
            if self.dim > 1:
                self.ang_stiffness = memory.root.init.angular_stiffness[:]
                self.inertia =  memory.root.init.inertia[:]
            else:
                self.ang_stiffness = np.zeros_like(self.force)
                self.inertia = np.zeros_like(self.force)
            self.failure_criterion = memory.root.init.failure_criterion[:]
            if self.dim > 3:
                self.tors_stiffness = memory.root.init.torsional_stiffness[:]
            else: self.tors_stiffness = np.zeros_like(self.force)
            if (self.alpha!=0).any():    
                self.alpha = memory.root.init.alpha[:]
            else:
                self.alpha = np.zeros_like(self.force)
            self.aspect_ratio =  memory.root.init.aspect_ratio[:]
            if isinstance(self.beam_section, np.ndarray):
                self.beam_section =  memory.root.init.beam_section[:]
            memory.close()
            
            print('Deleting all memory except init')
            memory = tb.open_file(self.mempath, 'w')
            group = memory.create_group('/', 'init')
            memory.create_carray(group, 'elements',
                                      obj=self.elems[:,:2].astype(int))
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            if self.dim > 1:
                memory.create_carray(group, 'angular_stiffness', 
                                          obj=self.ang_stiffness)
                memory.create_carray(group, 'inertia', 
                                          obj=self.inertia)
            if self.dim > 3:
                memory.create_carray(group, 'torsional_stiffness', 
                                          obj=self.tors_stiffness)
            if (self.alpha!=0).any(): 
                memory.create_carray(group, 'Timoshenko', 
                                          obj=self.alpha)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)
            memory.create_carray(group, 'aspect_ratio', 
                                      obj=self.aspect_ratio)
            if isinstance(self.beam_section, np.ndarray):
                memory.create_carray(group, 'beam_section', 
                                          obj=self.beam_section)  
            memory.close()
        else: print('Memory is deactivated.')
        pass

    def revert(self, step):
        """Go back to ``step`` in the memory of the element network."""
        if self.mempath != 'off':
            memory = tb.open_file(self.mempath, 'r')
            self.elems = memory.root.init.elements[:]
            self.del_elm = memory.root._v_children['iteration'+str(step)].deleted_elements[:]
            self.elems = np.delete(self.elems, self.del_elm, axis=0)
            self.displacement = memory.root._v_children['iteration'+str(step)].displacement[:]
            self.force = memory.root._v_children['iteration'+str(step)].force[:]
            self.failure_criterion = memory.root._v_children['iteration'+str(step)].failure_criterion[:]
            self.ax_stiffness = memory.root._v_children['iteration'+str(step)].axial_stiffness[:]
            if self.dim > 1:
                self.ang_stiffness = memory.root._v_children['iteration'+str(step)].angular_stiffness[:]
                self.inertia =  memory.root.init.inertia[:]
                self.inertia = np.delete(self.inertia, self.del_elm, axis=0)
            self.failure_criterion = memory.root._v_children['iteration'+str(step)].failure_criterion[:]
            if self.dim > 3:
                self.tors_stiffness = memory.root._v_children['iteration'+str(step)].torsional_stiffness[:]
            if (self.alpha!=0).any():    
                self.alpha = memory.root._v_children['iteration'+str(step)].Timoshenko[:]
            self.aspect_ratio =  memory.root.init.aspect_ratio[:]
            self.aspect_ratio = np.delete(self.aspect_ratio, self.del_elm, axis=0)
            if isinstance(self.beam_section, np.ndarray):
                self.beam_section =  memory.root.init.beam_section[:]
                self.beam_section = np.delete(self.beam_section, self.del_elm, axis=0)
            memory.close()
        else: print('Memory is deactivated.')
        pass
        
    def erase_tests(self, steps):
        """Erase the memory of events at given steps without changing the current state of the sample."""
        if self.mempath != 'off':
            try: len(steps)
            except: steps = np.array([steps])
            keys = ['iteration'+str(i) for i in steps]
            memory = tb.open_file(self.mempath, 'a')
            for k in keys:
                memory.remove_node('/', k, recursive=True)
            memory.close()
        else: print('Memory is deactivated.')
        pass
        
        
    def boundary_condition(self, nodes, displ, reset=False):
        """
        Establish the boundary conditions for a mechanical problem.
        
        Choose the index of the nodes you want to impose a specific 
        displacement (degree of freedom dependent) or to restrain a specific
        movement. If a node has already been defined, its former displacement 
        will be erased.

        Parameters
        ----------
        nodes : (Nnodes,) array-like
            Array containing the indices of the boundary nodes of interest.
        displ : (Nnodes, dof) or (dof,) numpy.ndarray
            Array containing the specific displacements to impose at the given 
            limit. NaN for no specific displacement, and 0 for anchoring. If
            displ.shape = (dof,), the same displacement is applied to all the 
            given nodes.
        reset : bool, optional
            Whether to create a new boundary table or to add additional 
            boundaries. The default is False.

        Returns
        -------
        Unodes : numpy.ndarray
            Full displacement of the nodes along all degrees of freedom. NaN 
            displacement is displacement that hasn't been precised, and 0 is 
            anchored node.

        """
        if displ.shape[0] not in [len(nodes), self.dof]:
            raise AttributeError("Displacement array needs to be of shape ("
                                 +str(self.dof)+",) or ("+str(len(nodes))
                                 +", "+str(self.dof)+"), not ("
                                 +str(displ.shape[0])+', '+str(displ.shape[1])
                                      +').')
        if len(displ.shape)==2 and (displ.shape[0]!=len(nodes)
                                    or displ.shape[1] != self.dof):
            raise AttributeError("Displacement array needs to be of shape ("+str(self.dof)+",) or ("+str(len(nodes))+", "+str(self.dof)+").")
            
        if displ.shape[0]==self.dof:
            Unodes = np.tile(displ, len(nodes)).reshape(1, len(nodes)*self.dof)
        else:
            Unodes = np.concatenate(displ).reshape(1, len(nodes)*self.dof)
            
        # Index in K_global array for easier inversion
        index = np.repeat(nodes, self.dof)
        for n in nodes:
            if len(index[index==n])==self.dof:
                index[index==n] = np.array([self.dof*n + i
                                            for i in range(self.dof)])
            else:
                index[index==n] = np.concatenate((
                    index[index==n][:-self.dof], 
                    np.array([self.dof*n + i for i in range(self.dof)])))
        if reset:
            self.boundaries = np.concatenate((
                index.reshape(1, len(nodes)*self.dof),
                Unodes), axis=0)
        else:
            self.boundaries = np.concatenate((
                self.boundaries,
                np.concatenate((index.reshape(1, len(nodes)*self.dof), 
                                Unodes), axis=0)),
                axis=1)

        # Cleaning doubles
        Iclear = []
        Ibounds = np.arange(self.boundaries.shape[1])
        for bound in np.unique(self.boundaries[0]):
            if len(Ibounds[self.boundaries[0]==bound])>1:
                Iclear.append(Ibounds[self.boundaries[0]==bound][:-1])
                
        if len(Iclear)>0:
            self.boundaries = np.delete(self.boundaries, np.concatenate(Iclear),
                                        axis=1)     
            
        # Deleting the unnecessary boundaries
        self.boundaries = np.delete(self.boundaries, 
                                    np.isnan(self.boundaries[1]), axis=1)
        
        return self.boundaries
    

    def apply_boundary(self, force=False):
        """
        Take the fixed displacement into account.

        Modify the global stiffness matrix K_global and the force F to
        account for the nodes with fixed displacement as stipulated in
        `boundaries`. The algorithm uses the Cholesky decomposition provided
        by sksparse.
        
        Parameters
        ----------
        false : bool, optional
            Whether the calculation is performed for applied displacement or
            applied force. The default is False, calculation is performed for
            applied displacement.

        Returns
        -------
        Elastic_Energy : FLOAT
            elastic energy
        Fnodes : numpy.ndarray
            Force applied on the nodes (should be 0 everywhere but at the boundaries)
        """
        
        Fnew = np.zeros(self.nodes.shape[0]*self.dof, dtype=float)
        Fnew[self.boundaries[0].astype(int)] = self.boundaries[1]
        
        if not force:
            # Taking the stiffness into account to truly obtain the displacement
            mask = np.ones((self.nodes.shape[0]*self.dof, 1))
            mask[self.boundaries[0].astype(int),0] = 0
            
    
            Fbound = self.K_global[:, self.boundaries[0].astype(int)]\
                @ sparse.diags_array([self.boundaries[1]], offsets=[0], format="lil")
            stored = sparse.csr_array(sparse.diags_array(
                mask.ravel(), format='lil') @ Fbound).sum(axis=1)
                
            Fnew = Fnew - np.asarray(stored).ravel()
        
        # New stiffness taking the loading into account
        Knew = deepcopy(self.K_global)
        
        if not force:
            Knew[self.boundaries[0].astype(int)] = 0
            Knew[:,self.boundaries[0].astype(int)] = 0
            Knew[self.boundaries[0].astype(int),
                 self.boundaries[0].astype(int)] = np.ones(self.boundaries.shape[1])
        
        Knew = sparse.csc_array(Knew)


        try:
            factor = cholesky(Knew)
            Unew = factor(Fnew)
        except: 
            print('Warning: The matrix appears to not be positive definite.'
                          + ' Switching to SuperLU algorithm.')
            factor = lng.splu(Knew)
            Unew = factor.solve(Fnew)

        # writing the displacements
        for i in range(self.dof):
            self.displacement[:,i] = Unew[i::self.dof]
            
        # axial force (or current for fusible-type element)
        if self.dof == 1:
            G = self.ax_stiffness
            I = G * (self.displacement[self.elems[:,1].astype(int), 0]\
                     - self.displacement[self.elems[:,0].astype(int), 0])
            self.force = I

        else:
            coord_in = self.nodes[self.elems[:,0].astype(int)]
            coord_out = self.nodes[self.elems[:,1].astype(int)]
            displacement_in = self.displacement[self.elems[:, 0].astype(int),
                                                :self.dim]
            displacement_out = self.displacement[self.elems[:, 1].astype(int),
                                                :self.dim]

            # linear development of the beam displacement
            delta_l = (coord_out - coord_in)\
                * (displacement_out - displacement_in)
            delta_l = np.sum(delta_l, axis=1) / self.elems[:, 2]

            # calculation of the axial force and the bending moment at 
            # barycentre
            N = -self.ax_stiffness * delta_l.reshape(self.elems.shape[0],1)
            self.force = N
                
        self.update_memory()
        self.Ntest += 1
        
        Elastic_Energy = 0.5 * np.transpose(Unew)\
            .dot(self.K_global.dot(Unew)) * 2
            
        Fnodes = self.K_global @ Unew
        
        self.K_global = sparse.lil_array(Knew)
        return Elastic_Energy, Fnodes
    
    def introduce_crack(self, x : float|np.ndarray, y : float|np.ndarray):
        """
        Introduce a crack in the sample by erasing elements.
        
        The crack can either be on the left side going to a given (x, y) position, between two x positions at a given y, or a rectangular crack.

        Parameters
        ----------
        x : float or array-like
            x or (x0, x1) positions of the crack. If array, the positions will be sorted.
        y : float or array-like
            y or (y0, y1) positions of the crack. If array, the positions will be sorted.
        """
        elmpos = np.concatenate((
            self.nodes[self.elems[:,0].astype(int)]\
            .reshape(self.elems.shape[0],self.dim,1),
            self.nodes[self.elems[:,1].astype(int)]\
            .reshape(self.elems.shape[0],self.dim,1)), axis=2)
            
        if isinstance(x, (np.ndarray, list)): x = sorted(x)
        if isinstance(y, (np.ndarray, list)): y = sorted(y)
            
        if not isinstance(x, (np.ndarray, list)):
            if isinstance(y, (np.ndarray, list)): y = y[0]
            I_cut = np.flatnonzero(
                np.prod(elmpos[:,0,:]<x, axis=1)
                    * ( (elmpos[:,1,1]<=y)*(elmpos[:,1,0]>=y)
                       + (elmpos[:,1,0]<=y)*(elmpos[:,1,1]>=y) )
                )
        else:
            if not isinstance(y, (np.ndarray, list)): 
                I_cut = np.flatnonzero(
                    np.prod(elmpos[:,0,:]<x[1], axis=1)
                    * np.prod(elmpos[:,0,:]>x[0], axis=1)\
                        * ( (elmpos[:,1,1]<=y)*(elmpos[:,1,0]>=y)
                           + (elmpos[:,1,0]<=y)*(elmpos[:,1,1]>=y) )
                    )
            else:
                I_cut = np.flatnonzero(
                    np.prod(elmpos[:,0,:]<x[1], axis=1)
                    * np.prod(elmpos[:,0,:]>x[0], axis=1)
                    * np.prod(elmpos[:,1,:]<x[1], axis=1)
                    * np.prod(elmpos[:,1,:]>x[0], axis=1)
                    )
        self.break_elem(I_cut)
        pass
    
    def break_elem(self, indx_elm):
        """Break the elems at indx_elm."""
        try: len(indx_elm)
        except: indx_elm = np.array([indx_elm])
        if not isinstance(indx_elm, np.ndarray): indx_elm = np.array(indx_elm)
        
        self.del_elm = np.concatenate((self.del_elm, indx_elm))
        
        self.crack = np.concatenate((
            self.crack,
            self.elems[indx_elm,:2].reshape(len(indx_elm),2)),
                axis=0)
            
        self.crack = np.delete(self.crack, 
                               np.prod(np.isnan(self.crack), axis=1, dtype=bool),
                               axis=0)
        self.last_broken['elems'] = self.elems[indx_elm]
        self.last_broken['angle'] = self.angle[indx_elm]
        self.last_broken['length'] = self.length[indx_elm]
        self.elems = np.delete(self.elems, indx_elm, axis=0)
        self.last_broken['force'] = self.force[indx_elm]
        self.force = np.delete(self.force, indx_elm, axis=0)
        self.last_broken['ax_stiffness'] = self.ax_stiffness[indx_elm]
        self.ax_stiffness = np.delete(self.ax_stiffness, indx_elm, axis=0)
        self.last_broken['ang_stiffness'] = self.ang_stiffness[indx_elm]
        self.ang_stiffness = np.delete(self.ang_stiffness, indx_elm, axis=0)
        self.last_broken['failure_criterion'] = self.failure_criterion[indx_elm]
        self.failure_criterion = np.delete(self.failure_criterion, indx_elm, 
                                           axis=0)
        self.last_broken['tors_stiffness'] = self.tors_stiffness[indx_elm]
        self.tors_stiffness = np.delete(self.tors_stiffness, indx_elm, axis=0)
        self.last_broken['alpha'] = self.alpha[indx_elm]
        self.alpha = np.delete(self.alpha, indx_elm, axis=0)
        self.last_broken['aspect_ratio'] = self.aspect_ratio[indx_elm]
        self.aspect_ratio = np.delete(self.aspect_ratio, indx_elm, axis=0)
        if isinstance(self.beam_section, np.ndarray):
            self.last_broken['beam_section'] = self.beam_section[indx_elm]
            self.beam_section = np.delete(self.beam_section, indx_elm, axis=0)
        self.last_broken['inertia'] = self.inertia[indx_elm]
        self.inertia = np.delete(self.inertia, indx_elm, axis=0)
            
    def shear_bend(self):
        """
        Extract the maximum moment and shear at the extremities of the beams.

        Compute for each beams max(abs(Mi), abs(Mj)) where i and j are the
        nodes at each end of the beam and V the shear force for 2D or 3D 
        case.

        Returns
        -------
        M : ARRAY OF FLOAT
            max(abs(Mi), abs(Mj)) for every element ij in the lattice (in the same order as the elems array).
        V : ARRAY OF FLOAT OR TUPLE
            shear force(s) for every element in the lattice (in the same order as the elems array).
            In 2D it is an array of float, in 3D it is an array of tuple (V1, V2) for each beam.
        """
        return shear_bend(self)
                        
    def __repr__(self):
        return str(self.dim)+'D '+str(self.type)+'-type element network of '+\
            str(self.nodes.shape[0])+' nodes and '+str(self.elems.shape[0])\
                +' elements with the memory of '+str(self.Ntest-1) +' tests.'

#%%% Specific element-types

class Fuse(Element):
    """
    Specific class of Element for random fuse network models.
    
    Parameters
    ----------
    lattice : treillis.Lattice
        lattice.
    Ic : numpy.ndarray
        array of the breaking current for each element, I.E. the failure criterion.
    sigma : FLOAT
        conductivity, I.E. the surfacic axial stiffness.    
        
    See Also
    --------
    treillis.mechanics.elements.Element
    treillis.mechanics.elements.Spring
    treillis.mechanics.elements.Beam
    
    Attributes
    ----------
    alpha : (Nelems,1) numpy.ndarray
        Corrective term in case of Timoshenko beams, otherwise 0-array
    angle : (Nelems, Ndim-1) np.ndarray
        Angle of the elements, polar angle if 2D, or 3D angles.
    ang_stiffness : (Nelems,1) numpy.ndarray
        Angular stiffness of the elements (0-array if dof<3)
    aspect_ratio : (Nelems,1) numpy.ndarray
        Aspect ratio of the elements gotten from the lattice
    ax_stiffness : (Nelems,1) numpy.ndarray
        Axial stiffness of the elements
    beam_section : numpy.ndarray or str
        Section of the elements gotten from the lattice
    boundaries : (N boundary nodes, dof, 2) numpy.ndarray
        Array containing the boundary nodes and their imposed displacement. Boundary nodes are anchored for an inf displacement.
    crack : (N cracked element, 2) numpy.ndarray
        Array containing the node indices around the crack.
    del_elm : numpy.ndarray
        Array containing the indices of the deleted elements with th ecrack progression.
    dim : int
        Dimensionality of the lattice
    displacement : (N nodes, dof) numpy.ndarray
        Nodes displacement. If dof>3, includes also the rotations.
    dof : int
        Degrees of freedom of the system
    elems : numpy.ndarray
        Elements as defined by the nodes indices and the length.
    failure_criterion : (Nelems,1) numpy.ndarray
        (N elems, 1) array containing the values to check whether or not an element is broken.
        TODO: find some way to implement it meaningfully.
    fixed_nodes : (N fixed,) numpy.ndarray
        Array containing the indices of the boundary nodes of the original lattice.
    K_global : (N nodes, N nodes) scipy.sparse.lil_array
        Global stiffness of the network. 
    L : float
        Average length of the lattice elements
    length : (Nelems,1) np.ndarray
        Length of each element.
    mempath : str
        Location of the h5 memory file of the test.
    nodes : (N nodes, N nodes) numpy.ndarray
        Position of the network node.
    Ntest : int
        Number of tests that have been performed.
    section_area : (Nelems,1) numpy.ndarray
        Section area of the lattice elements.
    force : (Nelems,1) numpy.ndarray
        array containing the line tension in the network elements
    tors_stiffness : (Nelems,1) numpy.ndarray
        array containing the torsional stiffness of the elements. 0-array for dof<=3
    type : str
        Type of Element network
    """
    
    def __init__(self, lattice: treillis.Lattice, Ic: float|np.ndarray,
                 sigma: float, location: str = None):
        
        if lattice.dim ==3:
            raise AttributeError('3D lattice provided. Fuse networks can only be 2D.')
        
        self.type = 'Fuse'
        self.dof = 1
        Element.__init__(self, lattice, self.dof, location)

        sigma_x_S = sigma * self.section_area

        self.displacement = np.zeros((self.nodes.shape[0], self.dof))

        self.force = np.zeros(self.elems.shape[0])
        
        if not isinstance(Ic, np.ndarray):
            self.failure_criterion = np.full((self.elems.shape[0], 1), Ic, 
                                             dtype=float)
        else: self.failure_criterion = deepcopy(Ic)
            
        self.ax_stiffness = sigma_x_S / self.elems[:, 2]
        
        print('Assembling global stiffness matrix')
        K_global = sparse.lil_array(
            (self.dof * self.nodes.shape[0], 
             self.dof * self.nodes.shape[0]),
            dtype=float)
        
        K_e = stiffness.fuse()
        K_local = np.stack(
            [(ast * K_e) for ast in self.ax_stiffness],
            axis=-1)
        
        K_global[self.elems[:,0].astype(int), self.elems[:,1].astype(int)]\
            += K_local[0,1,:]
            
        K_global[self.elems[:,1].astype(int), self.elems[:,0].astype(int)]\
            += K_local[1,0,:]
            
        e0 = np.unique(self.elems[:,0])
        e1 = np.unique(self.elems[:,1])
        
        K_global[e0.astype(int), e0.astype(int)]\
            += np.array([K_local[0,0,self.elems[:,0]==e].sum(axis=-1)
                           for e in e0])
            
        K_global[e1.astype(int), e1.astype(int)]\
            += np.array([K_local[1,1,self.elems[:,1]==e].sum(axis=-1)
                           for e in e1])
            
        self.K_global = K_global
        
        if location != 'off':
            if location is None: location = os.getcwd()
            self.mempath = os.path.abspath(location + r'/' + 'memory.h5')
            print('Memory will be saved at', self.mempath)
            memory = tb.open_file(self.mempath, 'w', driver='H5FD_CORE')
            
            group = memory.create_group('/', 'init')
            memory.create_carray(group, 'elements',
                                      obj=self.elems[:,:2].astype(int))
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)
            memory.create_carray(group, 'aspect_ratio', 
                                      obj=self.aspect_ratio)
            if isinstance(self.beam_section, np.ndarray):
                memory.create_carray(group, 'beam_section', 
                                          obj=self.beam_section)
            memory.close()
        else: self.mempath = 'off'

    def reassemble(self):
        """Recalculate the global stiffness."""
        self.Ntest += 1
        
        K_global = sparse.lil_array(
            (self.dof * self.nodes.shape[0], 
             self.dof * self.nodes.shape[0]),
            dtype=float)

        K_e = stiffness.fuse()
        K_local = np.stack(
            [(ast * K_e) for ast in self.ax_stiffness],
            axis=-1)
        
        K_global[self.elems[:,0].astype(int), self.elems[:,1].astype(int)]\
            += K_local[0,1,:]
            
        K_global[self.elems[:,1].astype(int), self.elems[:,0].astype(int)]\
            += K_local[1,0,:]
            
        e0 = np.unique(self.elems[:,0])
        e1 = np.unique(self.elems[:,1])
        
        K_global[e0.astype(int), e0.astype(int)]\
            += np.array([K_local[0,0,self.elems[:,0]==e].sum(axis=-1)
                           for e in e0])
            
        K_global[e1.astype(int), e1.astype(int)]\
            += np.array([K_local[1,1,self.elems[:,1]==e].sum(axis=-1)
                           for e in e1])
            
        self.K_global = sparse.lil_array(K_global)
        
        self.update_memory()
            
    def stress_strain(self):
        """
        Calculate the Local and global stress and strain tensors.
        
        The calculation is performed on every Voronoi cell attached to every 
        node of the lattice besides the fixed nodes and the crack. 

        Returns
        -------
        tensors : dict
            Dictionary containing the global and local tensors, as well as
            plotting tools and computation error.
            Dictionary items are:
                local stress : np.ndarray
                    local stress tensor.
                global stress : np.ndarray
                    global stress tensor.
                local strain : np.ndarray
                    Local strain tensor
                global strain : np.ndarray
                    Global strain tensor
                vertices : list
                    list of the coordinates of the vertices of each Voronoi 
                    cell.
                error : np.ndarray
                    Strain computation error.
        """
        new_vertices = []
        
        # Generate a Voronoi diagram around the nodes that are
        # not on the boundary, nor on the broken interface
        I_node = np.array([i for i in np.arange(self.nodes.shape[0])
                           if i not in self.fixed_nodes
                           and i not in self.crack.ravel()])
        vor = Voronoi(self.nodes)

        # local stress tensor computation:
        local_stress = np.zeros((0, self.dim, 1), dtype=float)
        vol_corr = np.zeros((0), dtype=float)
        
        local_strain = np.zeros((0, self.dim, 1), dtype=float)
        error = np.zeros((0, self.dim), dtype=float)
               
        for i in range(I_node.shape[0]):
            coord = np.nonzero(I_node[i] == self.elems[:, :2])
            contacts = self.elems[coord[0], 1 - coord[1]].astype(int)       
            
            # volume of each Voronoi cell
            ind = vor.point_region[I_node[i]]
            vol = ConvexHull(vor.vertices[vor.regions[ind]]).volume
            if self.dim == 2:
                vol *= self.L / np.mean(self.aspect_ratio[coord[0]])
                
            vol_corr = np.concatenate((vol_corr, np.array([vol])), axis=0)
            
            G = self.ax_stiffness[coord[0], 5]                
            dX = self.nodes[contacts] - self.nodes[I_node[i]]
            dU = self.displacement[contacts] - self.nodes[I_node[i]]
               
            # calculation of the current
            I = G * dU
            
            # calculation of local stress tensor
            value = np.sum(dX * I.reshape((len(contacts), 1)), axis=0)\
                / (2 * vol)
                    
            # calculation of local strain tensor
            value = np.sum((dX * dU.reshape(len(contacts), 1)), axis=0)\
                / np.sum((dX ** 2), axis=0)  
            Khi_2 = np.sum(((value * dX - dU.reshape(len(contacts), 1)) ** 2),
                           axis=0)
            error = np.concatenate((error, Khi_2[None,:]), axis=0)
            epsilon = np.concatenate((local_strain,
                                      value.reshape(1, self.dim, 1)), axis=0)
                
            # vertices for plot
            polygon = vor.vertices[vor.regions[ind]]
            new_vertices.append(polygon)
                
            local_stress = np.concatenate((local_stress, value.reshape(1, self.dim, 1)), 
                                   axis=0)
        
        # global stress tensor computation
        vol_tot = np.sum(vol_corr)
        global_stress = local_stress * vol_corr.reshape(len(vol_corr), 1, 1)\
            / vol_tot
        global_stress = np.sum(global_stress, axis=0)
        
        # global strain tensor computation
        global_strain = epsilon * vol_corr.reshape(len(vol_corr), 1, 1)\
            / vol_tot
        global_strain = np.sum(global_strain, axis=0)
        
        epsilon = 0.5 * epsilon
        global_strain = 0.5 * global_strain
        
        tensors = {
            'local stress': local_stress,
            'global stress': global_stress,
            'local strain': local_strain,
            'global strain': global_strain,
            'vertices': new_vertices,
            'error': error}
                
        return tensors
        

class Spring(Element):
    """
    Specific class of Element for linear spring network models.

    Parameters
    ----------
    lattice : treillis.Lattice
        lattice.
    sigma_c : numpy.ndarray or float
        array of the breakdown stress for each element.
    E : FLOAT
        Young's modulus of the bulk constitutive material.
    nu : FLOAT
        Poisson ratio of the bulk constitutive material.
    Timoshenko : bool
        add a corrective term based on Timoshenko beam model
        
    See Also
    --------
    treillis.mechanics.elements.Element
    treillis.mechanics.elements.Fuse
    treillis.mechanics.elements.Beam
    
    Attributes
    ----------
    alpha : (Nelems,1) numpy.ndarray
        Corrective term in case of Timoshenko beams, otherwise 0-array
    angle : (Nelems, Ndim-1) np.ndarray
        Angle of the elements, polar angle if 2D, or 3D angles.
    ang_stiffness : (Nelems,1) numpy.ndarray
        Angular stiffness of the elements (0-array if dof<3)
    aspect_ratio : (Nelems,1) numpy.ndarray
        Aspect ratio of the elements gotten from the lattice
    ax_stiffness : (Nelems,1) numpy.ndarray
        Axial stiffness of the elements
    beam_section : numpy.ndarray or str
        Section of the elements gotten from the lattice
    boundaries : (N boundary nodes, dof, 2) numpy.ndarray
        Array containing the boundary nodes and their imposed displacement. Boundary nodes are anchored for an inf displacement.
    crack : (N cracked element, 2) numpy.ndarray
        Array containing the node indices around the crack.
    del_elm : numpy.ndarray
        Array containing the indices of the deleted elements with th ecrack progression.
    dim : int
        Dimensionality of the lattice
    displacement : (N nodes, dof) numpy.ndarray
        Nodes displacement. If dof>3, includes also the rotations.
    dof : int
        Degrees of freedom of the system
    elems : numpy.ndarray
        Elements as defined by the nodes indices and the length.
    failure_criterion : (Nelems,1) numpy.ndarray
        (N elems, 1) array containing the values to check whether or not an element is broken.
        TODO: find some way to implement it meaningfully.
    fixed_nodes : (N fixed,) numpy.ndarray
        Array containing the indices of the boundary nodes of the original lattice.
    K_global : (N nodes, N nodes) scipy.sparse.lil_array
        Global stiffness of the network. 
    L : float
        Average length of the lattice elements
    length : (Nelems,1) np.ndarray
        Length of each element.
    mempath : str
        Location of the h5 memory file of the test.
    nodes : (N nodes, N nodes) numpy.ndarray
        Position of the network node.
    Ntest : int
        Number of tests that have been performed.
    section_area : (Nelems,1) numpy.ndarray
        Section area of the lattice elements.
    force : (Nelems,1) numpy.ndarray
        array containing the line tension in the network elements
    tors_stiffness : (Nelems,1) numpy.ndarray
        array containing the torsional stiffness of the elements. 0-array for dof<=3
    type : str
        Type of Element network
    """
    
    def __init__(self, lattice: treillis.Lattice, sigma_c: float|np.ndarray,
                 E: float, nu: float, Timoshenko: bool = True, location: str = None):
        self.inertia = None
        self.type = 'Spring'
        self.dof = lattice.dim
        Element.__init__(self, lattice, self.dof, location)

        if self.beam_section == "Square":
            I = ((self.L / self.aspect_ratio) ** 4) / 12  # for square beam
            # corrective term for Timoshenko's beam
            if Timoshenko:
                self.alpha = 2 * 6/5 * (1 + nu) / \
                    (self.aspect_ratio * self.elems[:, 2] / self.L) ** 2
                self.inertia = I
                    
        if self.beam_section == "Circular":
            I = (np.pi / 64) * (self.L / self.aspect_ratio) ** 4
            # corrective term for Timoshenko's beam
            if Timoshenko:
                self.alpha = 1.5 * 10/9 * (1 + nu) / \
                (self.aspect_ratio * self.elems[:, 2] / self.L) ** 2
                self.inertia = I

        if isinstance(self.beam_section, np.ndarray):
            I = np.array([inertia(self.beam_section[:,:,i], 
                                  self.L/self.aspect_ratio[i])[1]
                          for i in range(self.elems.shape[0])])
            if Timoshenko:
                equiv_diam = (I / (np.pi/64)) ** (1/4)
                self.alpha = 1.5 * 10/9 * (1 + nu) / \
                    (equiv_diam * self.elems[:, 2]) ** 2
                self.inertia = I
                
        self.displacement = np.zeros((self.nodes.shape[0], self.dof))

        self.force = np.zeros(self.elems.shape[0])
        
        if not isinstance(sigma_c, np.ndarray):
            self.failure_criterion = np.full((self.elems.shape[0], 1), sigma_c, 
                                             dtype=float)
        else: self.failure_criterion = deepcopy(sigma_c)
        
        self.ax_stiffness = np.reshape(np.full(
            self.elems.shape[0], E * self.section_area, dtype=float) /
                               lattice.elems[:, 2], (self.elems.shape[0], 1))
        self.ang_stiffness = np.reshape(E*I, (len(I),1))
        
        print('Assembling global stiffness matrix')
        if not Timoshenko:
            self.alpha = np.zeros(self.alpha.shape)

        K_e1 = stiffness.spring_1(self.dim)

        K_e2 = np.stack([stiffness.spring_2(self.dim, a)
                         for a in self.alpha], axis=-1)
        
        P_loc_glob = np.zeros((2 * self.dof, 2 * self.dof, self.elems.shape[0]),
                              dtype=float)
        Nrange = np.arange(self.elems.shape[0])
        theta = self.angle

        if self.dim ==2:
            P = np.zeros((2, 2, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,0])
            P[1,0,Nrange] = -np.sin(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,0])
        else:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.sin(theta[:,1]) * np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,1]) * np.sin(theta[:,0])
            P[0,2,Nrange] = np.cos(theta[:,1])
            P[1,0,Nrange] = np.cos(theta[:,1]) * np.cos(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,1]) * np.sin(theta[:,0])
            P[1,2,Nrange] = -np.sin(theta[:,1])
            P[2,0,Nrange] = -np.sin(theta[:,0])
            P[2,1,Nrange] = np.cos(theta[:,0])
            P[2,2,Nrange] = 0
            
        for k in range(2):
            P_loc_glob[self.dim * k:self.dim * (k + 1), 
                       self.dim * k:self.dim * (k + 1), :] = P
            
        K = np.stack([K_e1*a for a in self.ax_stiffness], axis=-1)
        K += self.ang_stiffness.reshape(1,1,self.elems.shape[0]) * K_e2

        K_local = np.stack([
            (P_loc_glob[:,:,i].T).dot(K[:,:,i].dot(P_loc_glob[:,:,i]))
            for i in Nrange], axis=-1)

        e0 = np.unique(self.elems[:,0]).astype(int)
        e1 = np.unique(self.elems[:,1]).astype(int)
        
        I, J = np.meshgrid(np.arange(self.dof), np.arange(self.dof),
                          indexing='ij')
        I = I.flatten()
        J = J.flatten()
        
        for (i,j) in zip(I,J):
            self.K_global[np.array([self.dof*e + i for e in e0]),
                          np.array([self.dof*e + j for e in e0])]\
                += np.array([sum(K_local[i, j, self.elems[:,0]==e])
                             for e in e0])
                
            self.K_global[np.array([self.dof*e + i for e in e1]),
                          np.array([self.dof*e + j for e in e1])]\
                += np.array([sum(K_local[self.dof + i, self.dof + j,
                                     self.elems[:,1]==e])
                             for e in e1])
                
            self.K_global[self.dof*self.elems[:,0] + i,
                          self.dof*self.elems[:,1] + j]\
                += np.array([K_local[i, self.dof + j, n]
                             for n in Nrange])
                
            self.K_global[self.dof*self.elems[:,1] + i,
                          self.dof*self.elems[:,0] + j]\
                += np.array([K_local[self.dof + i, j, n]
                             for n in Nrange])

        if location != 'off':
            if location is None: location = os.getcwd()
            self.mempath = os.path.abspath(location + r'/' + 'memory.h5')
            print('Memory will be saved at', self.mempath)
            memory = tb.open_file(self.mempath, 'w', driver='H5FD_CORE')
           
            group = memory.create_group('/', 'init')
            memory.create_carray(group, 'elements',
                                      obj=self.elems[:,:2].astype(int))
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            memory.create_carray(group, 'angular_stiffness', 
                                      obj=self.ang_stiffness)
            if Timoshenko:
                memory.create_carray(group, 'Timoshenko', 
                                          obj=self.alpha)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)
            memory.create_carray(group, 'aspect_ratio', 
                                      obj=self.aspect_ratio)
            if isinstance(self.beam_section, np.ndarray):
                memory.create_carray(group, 'beam_section', 
                                          obj=self.beam_section)
            memory.create_carray(group, 'inertia', 
                                      obj=self.inertia)
            memory.close()
        else: self.mempath = 'off'
        
    
    def reassemble(self):
        """Recalculate the global stiffness."""
        self.K_global = sparse.lil_array(
            (self.dof * self.nodes.shape[0], 
             self.dof * self.nodes.shape[0]),
            dtype=float)
        
        K_e1 = stiffness.spring_1(self.dim)

        K_e2 = np.stack([stiffness.spring_2(self.dim, a)
                         for a in self.alpha], axis=-1)
        
        P_loc_glob = np.zeros((2 * self.dof, 2 * self.dof, self.elems.shape[0]),
                              dtype=float)
        Nrange = np.arange(self.elems.shape[0])
        theta = self.angle

        if self.dim ==2:
            P = np.zeros((2, 2, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,0])
            P[1,0,Nrange] = -np.sin(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,0])
        else:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.sin(theta[:,1]) * np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,1]) * np.sin(theta[:,0])
            P[0,2,Nrange] = np.cos(theta[:,1])
            P[1,0,Nrange] = np.cos(theta[:,1]) * np.cos(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,1]) * np.sin(theta[:,0])
            P[1,2,Nrange] = -np.sin(theta[:,1])
            P[2,0,Nrange] = -np.sin(theta[:,0])
            P[2,1,Nrange] = np.cos(theta[:,0])
            P[2,2,Nrange] = 0
            
        for k in range(2):
            P_loc_glob[self.dim * k:self.dim * (k + 1), 
                       self.dim * k:self.dim * (k + 1), :] = P
            
        K = np.stack([K_e1*a for a in self.ax_stiffness], axis=-1)
        K += self.ang_stiffness.reshape(1,1,self.elems.shape[0]) * K_e2

        K_local = np.stack([
            (P_loc_glob[:,:,i].T).dot(K[:,:,i].dot(P_loc_glob[:,:,i]))
            for i in Nrange], axis=-1)

        e0 = np.unique(self.elems[:,0]).astype(int)
        e1 = np.unique(self.elems[:,1]).astype(int)
        
        I, J = np.meshgrid(np.arange(self.dof), np.arange(self.dof),
                          indexing='ij')
        I = I.flatten()
        J = J.flatten()
        
        for (i,j) in zip(I,J):
            self.K_global[np.array([self.dof*e + i for e in e0]),
                          np.array([self.dof*e + j for e in e0])]\
                += np.array([sum(K_local[i, j, self.elems[:,0]==e])
                             for e in e0])
                
            self.K_global[np.array([self.dof*e + i for e in e1]),
                          np.array([self.dof*e + j for e in e1])]\
                += np.array([sum(K_local[self.dof + i, self.dof + j,
                                     self.elems[:,1]==e])
                             for e in e1])
                
            self.K_global[self.dof*self.elems[:,0] + i,
                          self.dof*self.elems[:,1] + j]\
                += np.array([K_local[i, self.dof + j, n]
                             for n in Nrange])
                
            self.K_global[self.dof*self.elems[:,1] + i,
                          self.dof*self.elems[:,0] + j]\
                += np.array([K_local[self.dof + i, j, n]
                             for n in Nrange])
                
        self.update_memory()
        self.Ntest += 1
        
        
    def stress_strain(self):
        """
        Calculate the Local and global stress and strain tensors.
        
        The calculation is performed on every Voronoi cell attached to every 
        node of the lattice besides the fixed nodes and the crack. 

        Returns
        -------
        tensors : dict
            Dictionary containing the global and local tensors, as well as
            plotting tools and computation error.
            Dictionary items are:
                local stress : np.ndarray
                    local stress tensor.
                global stress : np.ndarray
                    global stress tensor.
                local strain : np.ndarray
                    Local strain tensor
                global strain : np.ndarray
                    Global strain tensor
                vertices : list
                    list of the coordinates of the vertices of each Voronoi 
                    cell.
                deformed vertices : list
                    List of the coordinates of vertices of the Voronoi cells
                    as defined on the deformed sample.
                error : np.ndarray
                    Strain computation error.
        """        
        I_node = np.array([i for i in np.arange(self.nodes.shape[0])
                           if i not in self.fixed_nodes
                           and i not in self.crack.ravel()])
        # nodesbroken = self.nodes[I_node]
        vor = Voronoi(self.nodes)
        vertices_undef = []
        
        dplt = self.displacement * np.where(np.isnan(self.displacement),
                                            np.zeros_like(self.displacement),
                                            np.ones_like(self.displacement))
        
        nodesdef = self.nodes + dplt
        vor_def = Voronoi(nodesdef)
        new_vertices_def = []
               
        # local stress tensor computation
        local_stress = np.zeros((0, self.dim, self.dim), dtype=float)
        vol_corr = np.zeros(0, dtype=float)
        
        # strain
        error = np.zeros(self.dim)
        local_strain = np.zeros((0, self.dim, self.dim), dtype=float)
        
        for I in range(I_node.shape[0]):
            i = I_node[I]
            coord = np.nonzero(i == self.elems[:, :2])
            contacts = self.elems[coord[0], 1 - coord[1]].astype(int)
                            
            # volume of each Voronoi cell
            ind = vor.point_region[i]
            vol = ConvexHull(vor.vertices[vor.regions[ind]]).volume
            
            if self.dim == 2:
                vol *= self.L / np.mean(self.aspect_ratio[coord[0]])
            
            vertices_undef += [vor.vertices[vor.regions[ind]]]    
            
            vol_corr = np.concatenate((vol_corr, np.array([vol])), axis=0)
            
            E_x_I = self.ang_stiffness[coord[0]].ravel()
            E_x_S_div_L = self.ax_stiffness[coord[0]].ravel()
            
            dX = self.nodes[contacts] - self.nodes[i].reshape(1, self.dim)
            dU = self.displacement[contacts]\
                - self.displacement[i].reshape(1, self.dim)
            L = self.length[coord[0]]
            phi = self.alpha[coord[0]]
            sig = np.zeros(self.dim ** 2, dtype=float)
            alpha = np.arctan2(dX[:, 1], dX[:, 0]) 
            
            P = np.array(
                [[np.cos(alpha), np.sin(alpha), np.zeros(len(contacts))],
                 [-np.sin(alpha), np.cos(alpha), np.zeros(len(contacts))],
                 [np.zeros(len(contacts)), np.zeros(len(contacts)), 
                  np.ones(len(contacts))]])
            
            if self.dim == 3:
                beta = np.arctan2(np.sqrt(np.sum(dX[:, :2] ** 2, axis=1)),
                                  dX[:, 2])
                alpha = np.concatenate((alpha.reshape(len(alpha), 1),
                                        beta.reshape(len(beta), 1)), axis=1)
                
                P = np.array(
                    [[np.sin(alpha[:, 1]) * np.cos(alpha[:, 0]),
                      np.sin(alpha[:, 1]) *  np.sin(alpha[:, 0]),
                      np.cos(alpha[:, 1])], 
                     [np.cos(alpha[:, 1]) * np.cos(alpha[:, 0]), 
                      np.cos(alpha[:, 1]) * np.sin(alpha[:, 0]),
                      -np.sin(alpha[:, 1])],
                     [-np.sin(alpha[:, 0]), np.cos(alpha[:, 0]), 
                      np.zeros(len(contacts))]])
      
            vect = P.T                                                                                      
    
            # linear development of the spring displacement
            delta_l = np.sum(vect[:, :, 0][:, :self.dim] * dU, axis=1)
            delta_v1 = np.sum(vect[:, :, 1][:, :self.dim] * dU, axis=1)
            
            if self.dim == 3:
                delta_v2 = np.sum(vect[:, :, 2][:, :self.dim]  * dU, axis=1)
                
            # calculation of the axial and shear strengths
            F = E_x_S_div_L * delta_l
            V1 = (12 * E_x_I * delta_v1 / L ** 3) / (1 + phi)
            F = np.concatenate((F.reshape(len(contacts), 1),
                                V1.reshape(len(contacts), 1)), axis=1)
            
            if self.dim == 3:
                V2 = (12 * E_x_I * delta_v2 / L ** 3) / (1 + phi)
                F = np.concatenate((F[:,None], V2[:,None]), axis=1)
                    
            # calculation of local stress tensor
            for k in range(self.dim ** 2):
                sig[k] = np.sum(
                    dX[:, k // self.dim]
                    * np.sum(F * vect[:, :, :self.dim][:, k % self.dim], 
                             axis=1)) / (2 * vol)
                                         
            local_stress = np.concatenate((local_stress, 
                                           sig.reshape(1, self.dim, self.dim)),
                                   axis=0)
            
            # vertices for plot
            ind = vor_def.point_region[i]
            polygon = vor_def.vertices[vor_def.regions[ind]]
            new_vertices_def.append(polygon)
            
            # strain
            eps, khi = globstrain_beam_spring(self.dim, self.nodes, 
                                              self.displacement, self.elems, i)
            
            local_strain = np.concatenate((local_strain, eps), axis=0)
            error += khi
        
        # global stress tensor computation:
        vol_tot = np.sum(vol_corr)
        global_stress = local_stress * vol_corr.reshape(len(vol_corr), 1, 1) / vol_tot
        global_stress = np.sum(global_stress, axis=0)
        
        # global strain
        global_strain = local_strain * vol_corr.reshape(len(vol_corr), 1, 1)\
            / vol_tot
        global_strain = np.sum(global_strain, axis=0)
    
        tensors = {
            'local stress': local_stress,
            'global stress': global_stress,
            'local strain': local_strain,
            'global strain': global_strain,
            'vertices': vertices_undef,
            'deformed vertices': new_vertices_def,
            'error': error}
    
        return tensors



class Beam(Element):
    """
    Specific class of Element for Timoshenko beam network models.

    Parameters
    ----------
    lattice : treillis.Lattice
        lattice.
    sigma_c : numpy.ndarray or float
        array of the breakdown stress for each element.
    E : FLOAT
        Young's modulus of the bulk constitutive material.
    nu : FLOAT
        Poisson ratio of the bulk constitutive material.
        
    See Also
    --------
    treillis.mechanics.elements.Fuse
    treillis.mechanics.elements.Spring
    treillis.mechanics.elements.Element
    
    Attributes
    ----------
    alpha : (Nelems,1) numpy.ndarray
        Corrective term in case of Timoshenko beams, otherwise 0-array
    angle : (Nelems, Ndim-1) np.ndarray
        Angle of the elements, polar angle if 2D, or 3D angles.
    ang_stiffness : (Nelems,1) numpy.ndarray
        Angular stiffness of the elements (0-array if dof<3)
    aspect_ratio : (Nelems,1) numpy.ndarray
        Aspect ratio of the elements gotten from the lattice
    ax_stiffness : (Nelems,1) numpy.ndarray
        Axial stiffness of the elements
    beam_section : numpy.ndarray or str
        Section of the elements gotten from the lattice
    boundaries : (N boundary nodes, dof, 2) numpy.ndarray
        Array containing the boundary nodes and their imposed displacement. Boundary nodes are anchored for an inf displacement.
    crack : (N cracked element, 2) numpy.ndarray
        Array containing the node indices around the crack.
    del_elm : numpy.ndarray
        Array containing the indices of the deleted elements with th ecrack progression.
    dim : int
        Dimensionality of the lattice
    displacement : (N nodes, dof) numpy.ndarray
        Nodes displacement. If dof>3, includes also the rotations.
    dof : int
        Degrees of freedom of the system
    elems : numpy.ndarray
        Elements as defined by the nodes indices and the length.
    failure_criterion : (Nelems,1) numpy.ndarray
        (N elems, 1) array containing the values to check whether or not an element is broken.
        TODO: find some way to implement it meaningfully.
    fixed_nodes : (N fixed,) numpy.ndarray
        Array containing the indices of the boundary nodes of the original lattice.
    K_global : (N nodes, N nodes) scipy.sparse.lil_array
        Global stiffness of the network. 
    L : float
        Average length of the lattice elements
    length : (Nelems,1) np.ndarray
        Length of each element.
    mempath : str
        Location of the h5 memory file of the test.
    nodes : (N nodes, N nodes) numpy.ndarray
        Position of the network node.
    Ntest : int
        Number of tests that have been performed.
    section_area : (Nelems,1) numpy.ndarray
        Section area of the lattice elements.
    force : (Nelems,1) numpy.ndarray
        array containing the line tension in the network elements
    tors_stiffness : (Nelems,1) numpy.ndarray
        array containing the torsional stiffness of the elements. 0-array for dof<=3
    type : str
        Type of Element network
    """
    
    def __init__(self, lattice: treillis.Lattice, sigma_c: np.ndarray|float,
                 E: float, nu: float, Timoshenko: bool = True, location: str = None):
        self.dof = 3 * (lattice.dim - 1)
        Element.__init__(self, lattice, self.dof, location)
        self.L = lattice.L
        self.type = 'Beam'
        if self.beam_section == "Square":
            I = ((self.L / self.aspect_ratio) ** 4) / 12  # for square beam
            S = (self.L / self.aspect_ratio)**2
            # corrective term for Timoshenko's beam
            self.alpha = 2 * 6/5 * (1 + nu) / \
                (self.aspect_ratio * self.elems[:, 2] / self.L) ** 2
                    
        if self.beam_section == "Circular":
            I = (np.pi / 64) * (self.L / self.aspect_ratio) ** 4
            S = np.pi * (lattice.L / (2 * self.aspect_ratio)) ** 2
            # corrective term for Timoshenko's beam
            self.alpha = 1.5 * 10/9 * (1 + nu) / \
            (self.aspect_ratio * self.elems[:, 2] / self.L) ** 2

        if isinstance(self.beam_section, np.ndarray):
            I = np.array([inertia(self.beam_section[:,:,i], 
                                  self.L/self.aspect_ratio[i])[1]
                          for i in range(self.elems.shape[0])])
            equiv_diam = (I / (np.pi/64)) ** (1/4)
            S = np.sum(self.beam_section) * (self.L/self.aspect_ratio\
                                             / self.beam_section.shape[0])**2
            self.alpha = 1.5 * 10/9 * (1 + nu) / \
                (equiv_diam * self.elems[:, 2]) ** 2
                
        self.inertia = I

        self.displacement = np.zeros((self.nodes.shape[0], self.dof))
        
        self.force = np.zeros(self.elems.shape[0])
        if not isinstance(sigma_c, np.ndarray):
            self.failure_criterion = np.full((self.elems.shape[0], 1), sigma_c, 
                                             dtype=float)
        else: self.failure_criterion = deepcopy(sigma_c)        
        self.ax_stiffness = np.reshape(E * S\
            / self.elems[:, 2], (self.elems.shape[0], 1))
        self.ang_stiffness = np.reshape(E*self.inertia, (self.elems.shape[0], 1))
        self.tors_stiffness = np.zeros(self.elems.shape[0])
        if self.dim == 3:
            G = 0.5 * E / (1 + nu)
            J_new = I * 2
            self.tors_stiffness = np.reshape(G * J_new / self.elems[:, 2], 
                                        (self.elems.shape[0], 1))
            
        print("Assembling global stiffness matrix")
        
        if not Timoshenko:
            self.alpha[:] = 0
            
        K_e1 = np.stack([stiffness.beam_1(self.dim) 
                         for _ in range(self.elems.shape[0])], axis=-1)
        if self.dim == 3:
            K_e5 = np.stack([stiffness.beam_5()
                             for _ in range(self.elems.shape[0])], axis=-1)
    
        K_e2 = np.stack([stiffness.beam_2(self.dim, a)
                         for a in self.alpha], axis=-1)
        K_e3 = np.stack([stiffness.beam_3(self.dim, a)
                         for a in self.alpha], axis=-1)
        K_e4 = np.stack([stiffness.beam_4(self.dim, a)
                         for a in self.alpha], axis=-1)
        
        P_loc_glob = np.zeros((2 * self.dof, 2 * self.dof, self.elems.shape[0]),
                              dtype=float)
        Nrange = np.arange(self.elems.shape[0])
        
        dd = self.nodes[self.elems[:,1].astype(int)]\
            - self.nodes[self.elems[:,0].astype(int)]
        angle = np.arctan2(dd[:,1], dd[:,0]).reshape(self.elems.shape[0])
        if self.dim == 3:
            angle = np.concatenate((
                angle.reshape(len(angle), 1), 
                np.arctan2(np.sqrt(np.sum(dd[:,:2]**2, axis=1)),
                           dd[:,2]).reshape(self.elems.shape[0],1)),
                axis=1)
        theta = angle.reshape(self.elems.shape[0], self.dim-1)

        if self.dim ==2:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,0])
            P[1,0,Nrange] = -np.sin(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,0])
            P[2,:,:] = 0
            P[:,2,:] = 0
            P[2,2,:] = 1
            
        else:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.sin(theta[:,1]) * np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,1]) * np.sin(theta[:,0])
            P[0,2,Nrange] = np.cos(theta[:,1])
            P[1,0,Nrange] = np.cos(theta[:,1]) * np.cos(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,1]) * np.sin(theta[:,0])
            P[1,2,Nrange] = -np.sin(theta[:,1])
            P[2,0,Nrange] = -np.sin(theta[:,0])
            P[2,1,Nrange] = np.cos(theta[:,0])
            P[2,2,Nrange] = 0
        
        for k in range(2 * (self.dim - 1)):
            P_loc_glob[3*k : 3*(k+1), 3*k : 3*(k+1), :] = P
        
        L = self.length.reshape(1,1,self.elems.shape[0])
        K = K_e2 / L**3 + K_e3 / L**2 + K_e4/L
        K *= self.ang_stiffness.reshape(1,1,self.elems.shape[0])
        K += self.ax_stiffness.reshape(1,1,self.elems.shape[0]) * K_e1
        if self.dim==3:
            K += self.tors_stiffness.reshape(1,1,self.elems.shape[0]) * K_e5
        
        K_local = np.stack([
            (P_loc_glob[:,:,i].T).dot(K[:,:,i].dot(P_loc_glob[:,:,i]))
            for i in Nrange], axis=-1)
        
        I, J = np.meshgrid(np.arange(self.dof), np.arange(self.dof),
                          indexing='ij')
        I = I.flatten()
        J = J.flatten()
         
        Kii, Koo, Kio, Koi = _glob(K_local, self.dof, 
                                   self.elems[:,:2].astype(int), 
                                   self.nodes.shape[0], I, J)
        
        e0 = np.unique(self.elems[:,0])
        e1 = np.unique(self.elems[:,1])

        for (i,j) in zip(I,J):
            self.K_global[np.array([self.dof*e + i for e in e0]),
                          np.array([self.dof*e + j for e in e0])] += Kii[i,j,:]
            
            self.K_global[np.array([self.dof*e + i for e in e1]),
                          np.array([self.dof*e + j for e in e1])] += Koo[i,j,:]
            
            self.K_global[self.dof*self.elems[:,0] + i,
                          self.dof*self.elems[:,1] + j] += Kio[i,j,:]
            
            self.K_global[self.dof*self.elems[:,1] + i,
                          self.dof*self.elems[:,0] + j] += Koi[i,j,:]
        
        if location != 'off':
            if location is None: location = os.getcwd()
            self.mempath = os.path.abspath(location + r'/' + 'memory.h5')
            print('Memory will be saved at', self.mempath)
            memory = tb.open_file(self.mempath, 'w', driver='H5FD_CORE')
            
            group = memory.create_group('/', 'init')
            memory.create_carray(group, 'elements',
                                      obj=self.elems[:,:2].astype(int))
            memory.create_carray(group, 'force', obj=self.force)
            memory.create_carray(group, 'axial_stiffness', 
                                      obj=self.ax_stiffness)
            memory.create_carray(group, 'angular_stiffness', 
                                      obj=self.ang_stiffness)
            memory.create_carray(group, 'torsional_stiffness', 
                                      obj=self.tors_stiffness)
            memory.create_carray(group, 'Timoshenko', 
                                      obj=self.alpha)
            memory.create_carray(group, 'failure_criterion', 
                                      obj=self.failure_criterion)
            memory.create_carray(group, 'aspect_ratio', 
                                      obj=self.aspect_ratio)
            if isinstance(self.beam_section, np.ndarray):
                memory.create_carray(group, 'beam_section', 
                                          obj=self.beam_section)
            memory.create_carray(group, 'inertia', 
                                      obj=self.inertia)
            memory.close()
        else: self.mempath = 'off'
    
    def reassemble(self):
        """Recalculate the global stiffness."""
        self.K_global = sparse.lil_array(
            (self.dof * self.nodes.shape[0], 
             self.dof * self.nodes.shape[0]),
            dtype=float)
        
        K_e1 = np.stack([stiffness.beam_1(self.dim) 
                         for _ in range(self.elems.shape[0])], axis=-1)
        if self.dim == 3:
            K_e5 = np.stack([stiffness.beam_5()
                             for _ in range(self.elems.shape[0])], axis=-1)
    
        K_e2 = np.stack([stiffness.beam_2(self.dim, a)
                         for a in self.alpha], axis=-1)
        K_e3 = np.stack([stiffness.beam_3(self.dim, a)
                         for a in self.alpha], axis=-1)
        K_e4 = np.stack([stiffness.beam_4(self.dim, a)
                         for a in self.alpha], axis=-1)
        
        P_loc_glob = np.zeros((2 * self.dof, 2 * self.dof, self.elems.shape[0]),
                              dtype=float)
        Nrange = np.arange(self.elems.shape[0])
        theta = self.angle

        if self.dim ==2:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,0])
            P[1,0,Nrange] = -np.sin(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,0])
            P[2,:,:] = 0
            P[:,2,:] = 0
            P[2,2,:] = 1
            
        else:
            P = np.zeros((3, 3, self.elems.shape[0]),
                         dtype=float)
            P[0,0,Nrange] = np.sin(theta[:,1]) * np.cos(theta[:,0])
            P[0,1,Nrange] = np.sin(theta[:,1]) * np.sin(theta[:,0])
            P[0,2,Nrange] = np.cos(theta[:,1])
            P[1,0,Nrange] = np.cos(theta[:,1]) * np.cos(theta[:,0])
            P[1,1,Nrange] = np.cos(theta[:,1]) * np.sin(theta[:,0])
            P[1,2,Nrange] = -np.sin(theta[:,1])
            P[2,0,Nrange] = -np.sin(theta[:,0])
            P[2,1,Nrange] = np.cos(theta[:,0])
            P[2,2,Nrange] = 0
        
        for k in range(2 * (self.dim - 1)):
            P_loc_glob[3*k : 3*(k+1), 3*k : 3*(k+1), :] = P
        
        L = self.length.reshape(1,1,self.elems.shape[0])
        K = K_e2 / L**3 + K_e3 / L**2 + K_e4/L
        K *= self.ang_stiffness.reshape(1,1,self.elems.shape[0])
        K += self.ax_stiffness.reshape(1,1,self.elems.shape[0]) * K_e1
        if self.dim==3:
            K += self.tors_stiffness.reshape(1,1,self.elems.shape[0]) * K_e5
        
        K_local = np.stack([
            (P_loc_glob[:,:,i].T).dot(K[:,:,i].dot(P_loc_glob[:,:,i]))
            for i in Nrange], axis=-1)
        
        I, J = np.meshgrid(np.arange(self.dof), np.arange(self.dof),
                          indexing='ij')
        I = I.flatten()
        J = J.flatten()

        Kii, Koo, Kio, Koi = _glob(K_local, self.dof, 
                                   self.elems[:,:2].astype(int), 
                                   self.nodes.shape[0], I, J)
        
        e0 = np.unique(self.elems[:,0])
        e1 = np.unique(self.elems[:,1])

        for (i,j) in zip(I,J):
            self.K_global[np.array([self.dof*e + i for e in e0]),
                          np.array([self.dof*e + j for e in e0])] += Kii[i,j,:]
            
            self.K_global[np.array([self.dof*e + i for e in e1]),
                          np.array([self.dof*e + j for e in e1])] += Koo[i,j,:]
            
            self.K_global[self.dof*self.elems[:,0] + i,
                          self.dof*self.elems[:,1] + j] += Kio[i,j,:]
            
            self.K_global[self.dof*self.elems[:,1] + i,
                          self.dof*self.elems[:,0] + j] += Koi[i,j,:]
                
        self.update_memory()
        self.Ntest += 1
        
    def stress_strain(self):
        """
        Calculate the Local and global stress and strain tensors.
        
        The calculation is performed on every Voronoi cell attached to every 
        node of the lattice besides the fixed nodes and the crack, using a 
        Viriel development. 

        Returns
        -------
        tensors : dict
            Dictionary containing the global and local tensors, as well as
            plotting tools and computation error.
            Dictionary items are:
                local stress : np.ndarray
                    local stress tensor.
                global stress : np.ndarray
                    global stress tensor.
                local strain : np.ndarray
                    Local strain tensor
                global strain : np.ndarray
                    Global strain tensor
                vertices : list
                    list of the coordinates of the vertices of each Voronoi 
                    cell.
                error : np.ndarray
                    Strain computation error.
        """              
        I_node = np.array([i for i in np.arange(self.nodes.shape[0])
                           if i not in self.fixed_nodes
                           and i not in self.crack.ravel()])
        # nodesbroken = self.nodes[I_node]
        vor = Voronoi(self.nodes)
        vertices_undef = []
        
        dplt = np.where(np.isnan(self.displacement),
                        np.zeros_like(self.displacement),
                        self.displacement)
        # print(dplt)
        nodesdef = self.nodes + dplt[:,:self.dim]
        vor_def = Voronoi(nodesdef)
        new_vertices_def = []
        
        # local stress tensor computation
        local_stress = np.zeros((0, self.dim, self.dim), dtype=float)
        vol_corr = np.zeros((0), dtype=float)
        
        # strain
        error = np.zeros(self.dim)
        local_strain = np.zeros((0, self.dim, self.dim), dtype=float)
        
        for I in range(I_node.shape[0]):
            i = I_node[I]
            coord = np.nonzero(i == self.elems[:, :2])
            contacts = self.elems[coord[0], 1 - coord[1]].astype(int)
                
            # volume of each Voronoi cell
            ind = vor.point_region[i]
            vol = ConvexHull(vor.vertices[vor.regions[ind]]).volume
            if self.dim == 2:
                vol *= np.mean(self.L / self.aspect_ratio[coord[0]])
            
            vertices_undef += [vor.vertices[vor.regions[ind]]]
            
            vol_corr = np.concatenate((vol_corr, np.array([vol])), axis=0)
            
            E_x_I = self.ang_stiffness[coord[0]].ravel()
            E_x_S_div_L = self.ax_stiffness[coord[0]].ravel()
            
            dX = self.nodes[contacts] - self.nodes[i].reshape(1, self.dim)
            dU = self.displacement[contacts, :self.dim]\
                - self.displacement[i, :self.dim].reshape(1, self.dim)
            L = self.length[coord[0]]
            phi = self.alpha[coord[0]]
            sig = np.zeros(self.dim ** 2, dtype=float)
            alpha = np.arctan2(dX[:, 1], dX[:, 0]) 
            
            P = np.array(
                [[np.cos(alpha), np.sin(alpha), np.zeros(len(contacts))],
                 [-np.sin(alpha), np.cos(alpha), np.zeros(len(contacts))],
                 [np.zeros(len(contacts)), np.zeros(len(contacts)), 
                  np.ones(len(contacts))]])
            
            if self.dim == 3:
                beta = np.arctan2(np.sqrt(np.sum(dX[:, :2] ** 2, axis=1)),
                                  dX[:, 2])
                alpha = np.concatenate((alpha.reshape(len(alpha), 1),
                                        beta.reshape(len(beta), 1)), axis=1)
                
                P = np.array(
                    [[np.sin(alpha[:, 1]) * np.cos(alpha[:, 0]),
                      np.sin(alpha[:, 1]) *  np.sin(alpha[:, 0]), 
                      np.cos(alpha[:, 1])],
                     [np.cos(alpha[:, 1]) * np.cos(alpha[:, 0]), 
                      np.cos(alpha[:, 1]) * np.sin(alpha[:, 0]),
                      -np.sin(alpha[:, 1])],
                     [-np.sin(alpha[:, 0]), np.cos(alpha[:, 0]),
                      np.zeros(len(contacts))]])
      
            vect = P.T                                                                                      
    
            # linear development of the beam displacement
            delta_l = np.sum(vect[:, :, 0][:, :self.dim] * dU, axis=1)
            delta_v1 = np.sum(vect[:, :, 1][:, :self.dim] * dU, axis=1)
            
            if self.dim == 3:
                delta_v2 = np.sum(vect[:, :, 2][:, :self.dim] * dU, axis=1)
                
            sum_theta = self.displacement[i, self.dim:]\
                + self.displacement[contacts, self.dim:]
                
            # calculation of the axial and shear forces
            F = E_x_S_div_L * delta_l
            V1 = (12 * E_x_I * delta_v1 / L ** 3 - 6 * E_x_I
                  * np.sum(vect[:, :, 2] * sum_theta, axis=1) / L ** 2)\
                / (1 + phi)
            F = np.concatenate((F.reshape(len(coord[0]), 1), 
                                V1.reshape(len(coord[0]), 1)), axis=1)
            
            if self.dim == 3:
                V2 = (12 * E_x_I * delta_v2 / L ** 3 + 6 * E_x_I * 
                      np.sum(vect[:, :, 1] * sum_theta, axis=1) / L ** 2)\
                    / (1 + phi)
                F = np.concatenate((F, V2.reshape(len(coord[0]), 1)), axis=1)
                    
            # calculation of local stress tensor
            for k in range(self.dim ** 2):
                sig[k] = np.sum(dX[:, k // self.dim]\
                                * np.sum(
                                    F * vect[:, :, :self.dim][:, k % self.dim],
                                         axis=1)) / (2 * vol)
                                         
            local_stress = np.concatenate((local_stress, 
                                           sig.reshape(1, self.dim, self.dim)),
                                          axis=0)
            
            ind = vor_def.point_region[i]
            polygon = vor_def.vertices[vor_def.regions[ind]]
            new_vertices_def.append(polygon)
            
            # strain
            eps, khi = globstrain_beam_spring(self.dim, self.nodes, 
                                              self.displacement, self.elems, i)
            
            local_strain = np.concatenate((local_strain, eps), axis=0)
            error += khi
        
        # global stress tensor computation
        vol_tot = np.sum(vol_corr)
        global_stress = local_stress * vol_corr.reshape(len(vol_corr), 1, 1)\
            / vol_tot
        global_stress = np.sum(global_stress, axis=0)
        
        # global strain
        global_strain = local_strain * vol_corr.reshape(len(vol_corr), 1, 1)\
            / vol_tot
        global_strain = np.sum(global_strain, axis=0)
    
        tensors = {
            'local stress': local_stress,
            'global stress': global_stress,
            'local strain': local_strain,
            'global strain': global_strain,
            'vertices': vertices_undef,
            'deformed vertices': new_vertices_def,
            'error': error}
          
        return tensors
   

# %% Specific functions to test the Element
@jit(parallel=True)
def _glob(local, dof, elems, nnodes, I, J):   
    Nrange = np.arange(elems.shape[0])
    
    e0 = np.unique(elems[:,0])
    e1 = np.unique(elems[:,1])
    
    Kii = np.zeros((dof, dof, len(e0)))
    Koo = np.zeros((dof, dof, len(e1)))
    Kio = np.zeros((dof, dof, elems.shape[0]))
    Koi = np.zeros((dof, dof, elems.shape[0]))
    
    for indx in prange(len(I)):
        i = I[indx]
        j = J[indx]
        
        to_add = np.zeros(e0.shape[0])
        for ei in prange(e0.shape[0]):
            to_add[ei] = sum(local[i, j, elems[:,0]==e0[ei]])
        Kii[i,j,:] = Kii[i,j,:] + to_add
            
        to_add = np.zeros(e1.shape[0])
        for ei in prange(e1.shape[0]):
            to_add[ei] = sum(local[dof + i, dof + j, elems[:,1]==e1[ei]])
        Koo[i,j,:] = Koo[i,j,:] + to_add
        
        kij = np.array([local[i, dof + j, n] for n in Nrange])
        Kio[i,j,:] += kij
            
        kji = np.array([local[dof + i, j, n] for n in Nrange])
        Koi[i,j,:] += kji
    return Kii, Koo, Kio, Koi

def shear_bend(network):
    """
    Extract the maximum moment and shear at the extremities of the beams.

    Compute for each beams max(abs(Mi), abs(Mj)) where i and j are the
    nodes at each end of the beam and V the shear force for 2D or 3D 
    case.

    Returns
    -------
    M : ARRAY OF FLOAT
        max(abs(Mi), abs(Mj)) for every element ij in the lattice (in the same order as the elems array).
    V : ARRAY OF FLOAT OR TUPLE
        shear force(s) for every element in the lattice (in the same order as the elems array).
        In 2D it is an array of float, in 3D it is an array of tuple (V1, V2) for each beam.
    """
    M = np.zeros((network.elems.shape[0],1))
    if network.dim==2:
        V = np.zeros((network.elems.shape[0],1))
    else:
        V = np.zeros((network.elems.shape[0],2))
    Phi = network.alpha

    for i in range(network.elems.shape[0]):
        V2 = 0
        E_x_I = network.ang_stiffness[i]
        dX = (network.nodes[network.elems[i, 1].astype(int)]\
              - network.nodes[network.elems[i, 0].astype(int)])
        dU = (network.displacement[network.elems[i, 1].astype(int),
                                   :network.dim]\
              - network.displacement[network.elems[i, 0].astype(int),
                                     :network.dim])
        L = network.elems[i, 2]
        phi = Phi[i]
        alpha = np.arctan2(dX[1], dX[0])

        P = np.array([[np.cos(alpha), np.sin(alpha), 0],
                      [-np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        if network.dim == 3:
            beta = np.arctan2(np.sqrt(np.sum(dX[:2] ** 2)), dX[2])
            alpha = [alpha, beta]
            P = np.array([[np.sin(alpha[1]) * np.cos(alpha[0]), 
                           np.sin(alpha[1]) * np.sin(alpha[0]), 
                           np.cos(alpha[1])],
                          [np.cos(alpha[1]) * np.cos(alpha[0]), 
                           np.cos(alpha[1]) * np.sin(alpha[0]),
                       -np.sin(alpha[1])],
                        [-np.sin(alpha[0]), np.cos(alpha[0]), 0]])

        vect = P.T
        # linear development of the beam displacement:
        theta2 = network.displacement[network.elems[i, 1].astype(int), 
                                      network.dim:]
        theta1 = network.displacement[network.elems[i, 0].astype(int),
                                   network.dim:]
        delta_v1 = np.sum(vect[:, 1][:network.dim] * dU)
        if network.dim == 3:
            delta_v2 = np.sum(vect[:, 2][:network.dim] * dU)

        Mi1 = (E_x_I / L ** 2)\
            * (L * ((4 + phi) * np.sum(vect[:, 2] * theta1)\
                    + (2 - phi) * np.sum(vect[:, 2] * theta2))\
               - 6 * delta_v1) / (1 + phi)
        Mj1 = (E_x_I / L ** 2)\
            * (L * ((2 - phi) * np.sum(vect[:, 2] * theta1)\
                    + (4 + phi) * np.sum(vect[:, 2] * theta2))\
               - 6 * delta_v1) / (1 + phi)
        V1 = -(12 * E_x_I * delta_v1 / L ** 3 - 6 * E_x_I\
               * np.sum(vect[:, 2] * (theta1 + theta2)) / L ** 2)\
            / (1 + phi)

        if network.dim == 3:
            V2 = -(12 * E_x_I * delta_v2 / L ** 3 + 6 * E_x_I\
                   * np.sum(vect[:, 1] * (theta1 + theta2)) / L ** 2)\
                / (1 + phi)

            Mi2 = (E_x_I / L ** 2)\
                * (L * ((4 + phi) * np.sum(vect[:, 1] * theta1)\
                        + (2 - phi) * np.sum(vect[:, 1] * theta2))\
                   + 6 * delta_v2) / (1 + phi)
            Mj2 = (E_x_I / L ** 2)\
                * (L * ((2 - phi) * np.sum(vect[:, 1] * theta1)\
                        + (4 + phi) * np.sum(vect[:, 1] * theta2))\
                   + 6 * delta_v2) / (1 + phi)
            Mi1 = np.sqrt(Mi1 ** 2 + Mi2 ** 2)
            Mj1 = np.sqrt(Mj1 ** 2 + Mj2 ** 2)

        if network.dim==2:
            V[i] = V1
        else:
            V[i,0] = V1
            V[i,1] = V2

        M[i] = max(abs(Mi1), abs(Mj1))

    return M, V


def globstrain_beam_spring(dim, nodes, displ, elems, i):
    # local stress tensor computation:
    coord = np.nonzero(i == elems[:, :2])
    contacts = elems[coord[0], 1 - coord[1]].astype(int)
    
    dX = nodes[contacts] - nodes[i].reshape(1, dim)
    dU = displ[contacts, :dim] - displ[i, :dim].reshape(1, dim)
    eps = np.zeros((dim, dim), dtype=float)
    
    # calculation of local strain tensor: 
    A = np.zeros((4 * (1 + dim // 3) + dim // 3,
                  4 * (1 + dim // 3) + dim // 3))
    B = np.zeros((4 * (1 + dim // 3) + dim // 3))    
    
    A[0, 0] = A[3, 3] = np.sum(dX[:, 0] ** 2)
    A[1, 2] = A[2, 1] = np.sum(dX[:, 1] ** 2)
    A[1, 0] = A[0, 2] = A[3, 1] = A[2, 3] = np.sum(dX[:, 0] * dX[:, 1])
    B[0] = np.sum(dX[:, 0] * dU[:, 0])
    B[3] = np.sum(dX[:, 0] * dU[:, 1])
    B[2] = np.sum(dX[:, 1] * dU[:, 1])
    B[1] = np.sum(dX[:, 1] * dU[:, 0])
    
    if dim == 3:
        A[4, 4] = A[5, 5] = A[7, 7] = np.sum(dX[:, 2] ** 2)
        A[6, 6] = A[0, 0]
        A[8, 8] = A[1, 2]
        A[0, 5] = A[3, 8] = A[4, 6] = A[5, 0] = A[6, 4] = A[7, 3]\
            = np.sum(dX[:, 0] * dX[:, 2])
        A[1, 5] = A[3, 7] = A[4, 8] = A[5, 2] = A[7, 1] = A[8, 4]\
            = np.sum(dX[:, 1] * dX[:, 2])
        A[8, 6] = A[6, 8] = A[1, 0]
        B[5] = np.sum(dX[:, 2] * dU[:, 0])
        B[7] = np.sum(dX[:, 2] * dU[:, 1])
        B[4] = np.sum(dX[:, 2] * dU[:, 2])
        B[6] = np.sum(dX[:, 0] * dU[:, 2])
        B[8] = np.sum(dX[:, 1] * dU[:, 2])
    
    C = np.linalg.solve(A, B)
    Khi_2 = np.zeros(dim)
    if dim == 2:
        eps = np.array([[C[0], (C[2] + C[3]) / 2],
                        [(C[2] + C[3]) / 2, C[1]]])
        Khi_2[0] += np.sum((C[0] * dX[:, 0] + C[2] * dX[:, 1] - dU[:, 0])
                           ** 2)
        Khi_2[1] += np.sum((C[1] * dX[:, 1] + C[3] * dX[:, 0] - dU[:, 1]) 
                           ** 2)
    else:
        eps = np.array([[C[0], (C[2] + C[3]) / 2, (C[5] + C[6]) / 2], 
                        [(C[2] + C[3]) / 2, C[1], (C[7] + C[8]) / 2], 
                        [(C[5] + C[6]) / 2, (C[7] + C[8]) / 2, C[4]]])
        Khi_2[0] += np.sum((C[0] * dX[:, 0] + C[2] * dX[:, 1] + C[5]\
                            * dX[:, 2] - dU[:, 0]) ** 2)
        Khi_2[1] += np.sum((C[1] * dX[:, 1] + C[3] * dX[:, 0] + C[7]\
                            * dX[:, 2] - dU[:, 1]) ** 2)
        Khi_2[2] += np.sum((C[4] * dX[:, 2] + C[6] * dX[:, 1] + C[8]\
                            * dX[:, 2] - dU[:, 2]) ** 2)
            
    return eps.reshape(1, dim, dim), Khi_2
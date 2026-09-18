"""
Import / Export
===============

Functions to import and export ``treillis.Lattice`` to bin or text files.
"""

#%% Package importation

import os

import numpy as np
import pandas as pd
import dill as pickle
import tables as tb

from .lattices import Lattice

__all__ = ['savebins', 'saveascii', 'saveh5', 'loadLattice']

LATPATH = os.path.abspath(os.path.dirname(__file__)
                          + r'/some_lattices/')

#%% Functions
#%%% Exporting

def savebins(path: str, lattice: Lattice):
    """Save the lattice as pickled Python bins."""
    print("Be aware that with updates, there may be compatibility issues with this lattice. It is recommended to save at least the `nodes` and `elems` tables as text as well.")
    path = os.path.abspath(path)
    
    # Making sure the path exists, or creating it
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    
    # Sauvegarder le tableau dans un fichier .pkl
    with open(path, 'wb') as file:
        pickle.dump(lattice, file)
        
    return print(f"The lattice has been saved in {path}")

def saveh5(path: str, lattice: Lattice):
    """Save the lattice as a HDF5 table."""
    path = os.path.abspath(path)
    
    # Making sure the path exists, or creating it
    os.makedirs(os.path.dirname(path) , exist_ok=True)
        
    class param(tb.IsDescription):
        L = tb.Float64Col()
        mesh = tb.StringCol(itemsize=100)
        section = tb.StringCol(itemsize=100)
    
    file = tb.open_file(path, 'w')
    table = file.create_table('/', 'parameters', param)
    p = table.row
    p['L'] = lattice.L
    p['mesh'] = lattice.type
    if isinstance(lattice.beam_section, str):
        p['section'] = lattice.beam_section
    else: p['section'] = 'array'
    p.append()
    table.flush()
    
    file.create_array('/', 'nodes', lattice.nodes)
    file.create_array('/', 'elements', lattice.elems)
    file.create_array('/', 'size', lattice.sizedom)
    file.create_array('/', 'ratio', lattice.aspect_ratio)
    file.create_array('/', 'fixed_nodes', lattice.fixed_nodes)
    
    if isinstance(lattice.beam_section, np.ndarray):
        file.create_carray('/', 'section', lattice.beam_section)
        
    file.close()
    return print(f"The lattice has been saved in {path}")

def saveascii(path: str, lattice: Lattice, *, bare: bool = True,
              zipped: bool = False, ext: str = 'csv', index_col : bool = False,
              start_index: int = 0):
    """
    Save the lattice as a collection of ASCII files.
    
    The lattice can be saved as either only the ``nodes`` and ``elems`` tables,
    or as all the parameters needed to generate a lattice, i.e. with the 
    addition of ``size``, ``length``, ``inimesh``,  ``aspect_ratio``,
    ``beam_section``, and ``fixed_nodes``.

    Parameters
    ----------
    path : str
        Path to directory where the files are saved.
    lattice : treillis.Lattice
        Lattice to be saved.
    bare : bool, optional
        Save the lattice as only ``nodes`` and ``elems`` if ``True``, and as all the parameters if ``False``. The default is ``True``.
    zipped: bool, optional
        Save the files in a compressed folder, default is ``False``.
    ext: str, optional
        File extension, default is csv
    index_col: bool, optional
        add an index column to the tables, default is False
    start_index: int, optional
        first index of the index column, default is 0

    Returns
    -------
    None.

    """
    path = os.path.abspath(path)
    os.makedirs(path , exist_ok=True)
    
    if index_col:
        elems = np.concatenate((
            np.arange(start_index, len(lattice.elems)+start_index).\
            reshape(len(lattice.elems),1), lattice.elems), axis=1)
        nodes = np.concatenate((
            np.arange(start_index, len(lattice.nodes)+start_index).\
            reshape(len(lattice.nodes),1), lattice.nodes), axis=1)
        if not bare:           
            ratio = np.concatenate((
                np.arange(start_index, len(lattice.aspect_ratio)+start_index).\
                reshape(len(lattice.aspect_ratio),1), lattice.aspect_ratio), 
                    axis=1)                          
    else:
        elems = lattice.elems
        nodes = lattice.nodes
        if not bare:           
            ratio = lattice.aspect_ratio
    
    if not bare:
        with open(os.path.abspath(path+r'\lattice_param.txt'), 'w') as file:
            file.write('size: ' + str(lattice.sizedom[0]) + ', '
                       + str(lattice.sizedom[1]))
            if lattice.sizedom.shape[0]>2:
                file.write(',' + str(lattice.sizedom[2]))
            if lattice.sizedom.shape[0]>3:
                file.write(',' + str(lattice.sizedom[3])
                           + str(lattice.sizedom[4]))
            file.write('\n')
            
            file.write('length: ' + str(lattice.L) + '\n')
            
            file.write('length: ' + str(lattice.type) + '\n')
        
            if isinstance(lattice.beam_section, str):
                file.write('section: ' + lattice.beam_section + '\n')
                
        if not isinstance(lattice.beam_section, str):
            np.savez_compressed(os.path.abspath(path+r'/sections.npz'),
                                [sec for sec in lattice.beam_section],
                                allow_pickle = False)            
        fixed_nodes = lattice.fixed_nodes
    
    if not zipped:
        np.savetxt(os.path.abspath(path+'/elems.'+ext), elems, delimiter=',')
        np.savetxt(os.path.abspath(path+'/nodes.'+ext), nodes, delimiter=',')
        if not bare:
            np.savetxt(os.path.abspath(path+r'/aspect_ratio.'+ext), ratio,
                       delimiter=',')
            np.savetxt(os.path.abspath(path+'/fixed_nodes.'+ext), fixed_nodes,
                       delimiter=',')
    else:
        if not bare:
            np.savez(os.path.abspath(path+r'/lattice'),
                     [nodes, elems, ratio, fixed_nodes], allow_pickle=False)
        else:
            np.savez(os.path.abspath(path+r'/lattice'),
                     [nodes, elems], allow_pickle=False)

    return print('The lattice has been saved.')

#%%% Importing

def loadLattice(filepath)-> Lattice:
    """
    Create a treillis lattice from pickled or ASCII files.
    
    if filepath == 'lattice_XX' with XX in [28, 167], will load one of the base pregenerated lattcies.

    Parameters
    ----------
    filepath: str or os.path
        path to the files or the folder containing the files to load.

    Returns
    -------
    lattice : treillis.Lattice
    """
    bases = ['lattice_'+str(i) for i in np.arange(28, 168, dtype=int)]
    if filepath in bases:
        return load_base(filepath)
    
    filepath = os.path.abspath(filepath)

    # load pickled file if pickled
    if '.pkl' in filepath:
        with open(filepath, 'rb') as file:
            lattice = pickle.load(file)
        return lattice
    
    # load h5
    if '.h5' in filepath:
        file = tb.open_file(filepath, 'r')
        nodes = file.root.nodes[:]
        elems = file.root.elements[:]
        size = file.root.size[:]
        if file.root.parameters.col('L').size > 0:
            length = None
        else:
            length = float(file.root.parameters.col('L')[0])
        if file.root.parameters.col('mesh').size > 0:
            inimesh = None
        else:
            inimesh = str(file.root.parameters.col('mesh')[0])
        aspect_ratio = file.root.ratio[:]
        fixed = file.root.fixed_nodes[:]
        if file.root.parameters.col('section')[0].astype(str) == 'array':
            section = file.root.section[:]
        else:
            section = file.root.parameters.col('section')[0].astype(str)
        return Lattice(nodes, elems, size=size, length=length, inimesh=inimesh,
                       aspect_ratio=aspect_ratio, beam_section=section,
                       fixed_nodes=fixed)
    
    # load bare zipped
    if '.npz' in filepath:
        nodes, elems = np.load(filepath)
        return Lattice(nodes, elems)
    
    # load full zipped
    list_file = os.listdir(filepath)
    
    if True in ['lattice_param' in file for file in list_file]:
        with open(os.path.abspath(filepath + '/'
                                  + '/lattice_param.txt'), 'r') as file:
            param = file.readlines()
        size = np.zeros(param[0].count(',')+1)
        i0 = param[0].index(':')+1
        i1 = param[0].index(',')
        for i in range(len(size)):
            size[i] = float(param[0][i0:i1])
            i0 = i1+1
            try: i1 += 2+param[0][i1+2:].index(',')
            except: i1 = param[0].index('\n')
            if i1<=i0: i1 = param[0].index('\n')
        length = float(param[1][param[1].index(':')+1:])
        inimesh = param[2][param[2].index(':')+2:-1]
        if len(param) == 4: sections = param[3][param[3].index(':')+2:-1]
    else: size, length, inimesh, sections = None, None, None, 'Circular'
        
    if True in ['section' in file for file in list_file]:
        i = ['section' in file for file in list_file].index(True)
        sections = np.load(os.path.abspath(filepath + '/' + list_file[i]),
                           allow_pickle=True)
    
    if 'lattice.npz' in list_file:
        try: nodes, elems, ratio, fixed_nodes = np.load(
            os.path.abspath(filepath + r'/lattice.npz'),
            allow_pickle=True)
        except ValueError: 
            nodes, elems = np.load(
            os.path.abspath(filepath + r'/lattice.npz'),
            allow_pickle=True)
            ratio, fixed_nodes = None, None
        return Lattice(nodes, elems, size=size, length=length, inimesh=inimesh,
                       aspect_ratio=ratio, beam_section=sections,
                       fixed_nodes=fixed_nodes)

    # load unzipped
    if True in ['elems' in file for file in list_file]:
        i = ['elems' in file for file in list_file].index(True)
        ext = list_file[i][list_file[i].index('.'):]
        elems = np.loadtxt(os.path.abspath(filepath + '/' + 'elems' + ext),
                           delimiter=',')
        nodes = np.loadtxt(os.path.abspath(filepath + '/' + 'nodes' + ext),
                           delimiter=',')
        
        # deleting the index column if present
        if (nodes[:,0]-nodes[0,0]==np.arange(nodes.shape[0])).all() :
            nodes = nodes[:,1:]
            elems = elems[:,1:]
            
        if 'aspect_ratio'+ext in list_file:
            ratio = np.loadtxt( os.path.abspath(filepath + '/' + 'aspect_ratio'\
                            + ext),
                                               delimiter=',')
            fixed = np.loadtxt(os.path.abspath(filepath + '/' + 'fixed_nodes'\
                            + ext),
                                               delimiter=',')
        else: ratio, fixed = 10., None
        
        return Lattice(nodes, elems, size=size, length=length, inimesh=inimesh,
                       aspect_ratio=ratio, beam_section=sections,
                       fixed_nodes=fixed)
    
    else: print('There might be an error with the directory, I cannot find the files.')

def load_base(name):
    """Load one of the lattices created by Antoine Montiel in 2019."""
    latindex=name.index('_')+1
    nodes = np.loadtxt(os.path.abspath(LATPATH + '/' + name[latindex:] 
                                       + '/nodes.csv'))
    elems = np.loadtxt(os.path.abspath(LATPATH + '/' + name[latindex:] 
                                       + '/elems.csv'))
    with open(os.path.abspath(LATPATH + '/' + name[latindex:]
                              + '/parameters.txt'), 'r') as file:
        param = file.readlines()
    size = np.zeros(param[0].count(',')+1)
    i0 = param[0].index('[')+1
    i1 = param[0].index(',')
    for i in range(len(size)):
        size[i] = float(param[0][i0:i1])
        i0 = i1+2
        try: i1 += 2+param[0][i1+2:].index(',')
        except: i1 = param[0].index(']')
        if i1<=i0: i1 = param[0].index(']')
    length = float(param[1][param[1].index('\t')+1:])
    mesh = param[2][param[2].index('\t')+1:-1]
    ratio = float(param[3][param[3].index('\t')+1:])
    section = param[4][param[4].index('\t')+1:]
    return Lattice(nodes, elems, size=size, length=length, inimesh=mesh, 
                   aspect_ratio=ratio, beam_section=section)
       
def last_pack(dim: int = 3):
    """Load the last bead packing generated as a (Nbeads, Ndim+1) numpy array whose last column is the bead size."""
    if dim == 3:
        file = os.path.abspath(os.path.dirname(__file__)
                               + r'/3D-packing-generation/packing.xyzd')
        pck = np.fromfile(file)
        pck = pck.reshape((int(len(pck)/4), 4))
        pck[:,:3] -= pck[:,:3].mean(axis=0)
        return pck
    else:
        file = os.path.abspath(os.path.dirname(__file__)
                               + r'/3D-packing-generation/packing.npy')
        pck = np.fromfile(file)
        return pck
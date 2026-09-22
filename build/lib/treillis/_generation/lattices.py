"""
Lattices
========

Generate the whole 2D or 3D lattice in a given domain. The domain is filled
with copies of the initial mesh, generated using symmetries.
The lattice stops growing when it reaches th elimits of the domain.
"""

#%% Package importation
from copy import deepcopy

import numpy as np
from numba import jit, prange, njit
from scipy.spatial import ConvexHull, Voronoi, Delaunay

TYPEFLOAT = np.float64
DECIMAL = 9

from .packings import makepacking, frompacking
from .meshes import call

__all__ = ['Domain', 'Lattice', 'Density', 'Density2', 'shake']

#%% Some functions

def Sym2D(nodes1, nodes2, nodes):
    """
    Line symmetry in 2D.

    The 2D coordinates of the symmetric of nodes are found with respect to
    a line defined by nodes1 and nodes2

    Parameters
    ----------
    **nodes1**: ARRAY OF FLOAT
        first node that defines the line.

    **nodes2**: ARRAY OF FLOAT
        second node that defines the line.

    **nodes**: ARRAY OF FLOAT
        node whose symmetric coordinates you want.

    Returns
    -------
    ARRAY OF FLOAT
        coordinates of the symmetric of nodes.

    """
    v = nodes2 - nodes1

    norme = np.sqrt(np.sum(v ** 2))
    BH = ((nodes[0] - nodes1[0]) * v[0] + (nodes[1] - nodes1[1]) * v[1]) / norme

    x_sym = 2 * nodes1[0] - nodes[0] + 2 * BH * v[0] / norme
    y_sym = 2 * nodes1[1] - nodes[1] + 2 * BH * v[1] / norme

    return np.array([x_sym, y_sym])


def Sym3D(nodes1, nodes2, nodes3, nodes):
    """
    Plane symmetry in 3D.

    Function to get the 3D coordinates of the symmetric of nodes with 
    respect to a plane defined by nodes1, nodes2 and nodes3.

    Parameters
    ----------
    nodes1 : ARRAY OF FLOAT
        first node that defines the plane.
    nodes2 : ARRAY OF FLOAT
        second node that defines the plane.
    nodes3 : ARRAY OF FLOAT
        third node that defines the plane.
    nodes : ARRAY OF FLOAT
        node whose symmetric coordinates you want.

    Returns
    -------
    ARRAY OF FLOAT
        coordinates of the symmetric of nodes.

    """
    u = nodes1
    v = nodes3 - nodes1
    w = nodes2 - nodes1
    A = w[1] * v[2] - w[2] * v[1]
    B = w[2] * v[0] - w[0] * v[2]
    C = w[0] * v[1] - w[1] * v[0]
    D = A * u[0] + B * u[1] + C * u[2]

    norme = A ** 2 + B ** 2 + C ** 2
    t = -(A * nodes[0] + B * nodes[1] + C * nodes[2] - D) / norme
    x_i = nodes[0] + A * t
    y_i = nodes[1] + B * t
    z_i = nodes[2] + C * t

    x_sym = 2 * x_i - nodes[0]
    y_sym = 2 * y_i - nodes[1]
    z_sym = 2 * z_i - nodes[2]

    return np.array([x_sym, y_sym, z_sym])


def Rot3D(plane, nodes):
    """
    Rotate around a point in 3D.

    Returns the cartesian coordinates of a point rotation of pi/3 around an 
    axis located at the center of the plane and normal to it.
    Method used for the truncated octahedron lattice.

    Parameters
    ----------
    plane : ARRAY OF FLOAT
        coordinates of the plane that defines the rotation axis.
    nodes : ARRAY OF FLOAT
        nodes you want to rotate.

    Returns
    -------
    nodes_rot : ARRAY OF FLOAT
        rotated nodes.

    """
    nodes_rot = np.zeros(np.shape(nodes), dtype=TYPEFLOAT)
    center = np.sum(plane, axis=0) / 6
    u = plane[0, :] - center
    v = plane[1, :] - center
    axe_rot =  np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0]])
    axe_rot *= 1 / (np.sqrt(np.sum(axe_rot ** 2)))

    a = axe_rot[0]
    b = axe_rot[1]
    c = axe_rot[2]
    A = np.array([[a ** 2, a * b, a *c], [b * a, b ** 2, b * c],
                  [c* a, c * b, c ** 2]])
    B = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    M = (1 - np.cos(np.pi / 3)) * A + np.cos(np.pi / 3) * \
    np.eye(3, dtype=TYPEFLOAT) + np.sin(np.pi / 3) * B

    for j in range(np.shape(nodes)[0]):
        nodes_rot[j, :] = np.dot((nodes[j, :] - center).reshape(1, 3), M).ravel()\
            + center

    return nodes_rot


def Density(lattice):
    """
    Calculate density with apparent box volume.

    Calculation of density using the apparent volume of the box and the
    total volume of elements considering a square or circular section for
    the beam. Only a small error is made for small domain.

    Parameters
    ----------
    lattice : Lattice
        lattice whose density you want to know.

    Returns
    -------
    density : float
        density of the lattice.
        
    See Also
    --------
    treillis.Density2
    
    Examples
    --------
    >>> import treillis
    >>> lat = treillis.loadLattice('lattice_130')
    >>> treillis.Density(lat)
    np.float64(0.05199386828508399)
    
    Notes
    -----
    Lattices have a ``density`` property implemented to get their density, which tries both
    Density2 and Density.

    """
    D = lattice.node_section
    dim = lattice.dim
    nodes = lattice.nodes
    elems = lattice.elems
    domain = lattice.domain
    sizedom = lattice.sizedom
    ar = lattice.aspect_ratio

    # calculation of volume of domain:
    if domain == 'ball' and dim == 2:
        app_size = np.max(nodes, axis=None)-np.min(nodes,axis=None)
        app_size/=2
        voldomain = np.pi * app_size ** 2

    elif domain == 'ball' and dim == 3:
        app_size = np.max(nodes, axis=None)-np.min(nodes,axis=None)
        app_size/=2
        voldomain = (4 / 3) * np.pi * app_size ** 3

    elif domain == 'rectangle' and dim == 2:
        voldomain = (np.max(nodes[:,0])-np.min(nodes[:,0])) *\
            (np.max(nodes[:,1])-np.min(nodes[:,1]))

    elif domain == 'rectangle' and dim == 3:
        voldomain = (np.max(nodes[:,0])-np.min(nodes[:,0])) *\
            (np.max(nodes[:,1])-np.min(nodes[:,1])) *\
                (np.max(nodes[:,2])-np.min(nodes[:,2]))

    elif domain == 'wedge' and dim == 2:
        large = (np.max(nodes[:,0])-np.min(nodes[:,0])) *\
            (np.max(nodes[:,1])-np.min(nodes[:,1]))

        small = sizedom[3]*sizedom[4]

        voldomain = large - small

    elif domain == 'wedge' and dim == 3:
        large = (np.max(nodes[:,0])-np.min(nodes[:,0])) *\
            (np.max(nodes[:,1])-np.min(nodes[:,1]))

        small = sizedom[3]*sizedom[4]

        voldomain = (large - small) *\
            (np.max(nodes[:,2])-np.min(nodes[:,2]))

    else:
        raise ValueError("Unknown domain shape, please implement it to calculate the density.")


    # calculation of total volume of matter in the beams but the nodes:
    if dim==2:
        L_beam = np.sqrt((nodes[elems[:,0].astype(int),0] -\
                          nodes[elems[:,1].astype(int),0])**2 +\
                         (nodes[elems[:,0].astype(int),1] -\
                          nodes[elems[:,1].astype(int),1])**2) - D
        volbeam =  np.sum(L_beam**2/ar)
    else:
        L_beam = np.sqrt((nodes[elems[:,0].astype(int),0] -\
                          nodes[elems[:,1].astype(int),0])**2 +\
                         (nodes[elems[:,0].astype(int),1] -\
                          nodes[elems[:,1].astype(int),1])**2 +\
                             (nodes[elems[:,0].astype(int),2] -\
                              nodes[elems[:,1].astype(int),2])**2) - D

        if lattice.beam_section == 'Square':
            volbeam = np.sum(L_beam**3/ar**2)

        elif lattice.beam_section == 'Circular':
            volbeam  = np.sum(L_beam**3 * np.pi /(4*ar**2))
        elif isinstance(lattice.beam_section, np.ndarray):
            volbeam = np.sum(L_beam * np.sum(lattice.beam_section)\
                             * (L_beam/ar / lattice.beam_section.shape[0])**2)
        else:
            raise ValueError('Beam section is not well defined.')

    volbeam += len(nodes) * 4/3 * np.pi * (D/2)**3

    density = volbeam / voldomain

    return density


def Density2(lattice):
    """
    Calculate density from Voronoi cell volume.

    Calculation of density from voronoi cell volume considering a square or
    circular section for the beam. Value calculated through the voronoi 
    cells volume.

    Parameters
    ----------
    lattice : LATTICE OBJECT
        lattice whose density you want to know.

    Returns
    -------
    density : FLOAT
        density of the lattice.
        
    See Also
    --------
    treillis.Density
    
    Examples
    --------
    >>> import treillis
    >>> lat = treillis.loadLattice('lattice_130')
    >>> treillis.Density2(lat)
    Calculating the fixed nodes
    np.float64(0.08405885209869692)
    
    Notes
    -----
    Lattices have a ``density`` property implemented to get their density, which tries both
    Density2 and Density.

    """
    volume = 0
    vol = 0
    vor = Voronoi(lattice.nodes)

    fixed = lattice.fixed_nodes
    
    @jit(parallel=True)
    def square(nodes, elems, fixed, node_section, dim, length):
        v = 0
        for i in prange(nodes.shape[0]):
            linked=np.nonzero(i==elems[:,:2])
            if i not in fixed:
                v += 0.5 * np.sum(length[linked[0]]
                                  - lattice.node_section)\
                    * (node_section) ** (dim - 1)
        return v
    
    @jit(parallel=True)
    def circ(nodes, elems, fixed, node_section, dim, length):
        v = 0
        for i in prange(nodes.shape[0]):
            linked=np.nonzero(i==elems[:,:2])
            if i not in fixed:
                v += 0.5 * np.sum(length[linked[0]]
                                  - node_section) * np.pi\
                    * (0.5 * node_section) ** 2
        return v
    
    @jit(parallel=True)
    def other(nodes, elems, fixed, node_section, beam_section, dim, length):
        v = 0
        for i in prange(nodes.shape[0]):
            linked=np.nonzero(i==elems[:,:2])
            if i not in fixed:
                v += .5 * np.sum((elems[linked[0], 2] - node_section)\
                    * (node_section / beam_section[linked[0]].shape[0])**2\
                        * np.sum(beam_section[linked[0]]))
        return v
    
    if lattice.beam_section == 'Square' or lattice.dim == 2:
        volume += square(lattice.nodes, lattice.elems, fixed,
                         lattice.node_section, lattice.dim, lattice.length)
    elif lattice.beam_section == 'Circular':
        volume += circ(lattice.nodes, lattice.elems, fixed,
                         lattice.node_section, lattice.dim, lattice.length)
    elif isinstance(lattice.beam_section, np.ndarray):
        volume += other(lattice.nodes, lattice.elems, fixed,
                         lattice.node_section, lattice.beam_section, 
                         lattice.dim, lattice.length)
    else:
        raise ValueError('Beam section is not well defined.')


    for i in range(lattice.nodes.shape[0]):
        # volume of each Voronoi cell:
        if i not in fixed:
            ind = vor.point_region[i]
            pts = vor.regions[ind]
            Vvor = ConvexHull(vor.vertices[pts]).volume
            vol += Vvor
            volume += Vvor * (lattice.node_section/lattice.L)**lattice.dim

    try:
        density = volume / vol
    except ZeroDivisionError:
        density = Density(lattice)
        print("Warning: the density calculated exactly with Voronoi failed. Density is approximated from the box.")

    return density

#%% Definition of the object domain to map

class Domain:
    """
    Box containing the lattice, which is used to define its general shape.

    General object to describe the Domain to mesh. Following the length of 
    the array size it detects which type of domain it is. In the array 
    size, the order is x, y, z (and possibly, for wedge geometry, slit 
    heigth and length)

    Parameters
    ----------
    size : ARRAY OF FLOAT or LIST
           it contains the dimension of the box.

           - if one value in the array => circle (2D) or ball (3D)
           - if 2 values => rectangle (2D, incompatible with 3D meshes) in 
           order (x, y)
           - if 3 values => 3D box in order (x, y, z)
           - if 5 values => wedge splitting geometry in the order
           (a, b, c, d, e) where a is in the x direction, b in the y 
           direction and c in the z direction
           
            .. warning::
                for 2D wedge splitting by convention you have to set a 
                value for c even if it is not used
                
                
            .. code-block:: python
                :force:
                    
                .........e
                .......<--->
                        ________________________
                .......|@@@@@@@@@@@@@@@@@@@@@@@@|...^
                ...^...|@@@@@@@@@@@@@@@@@@@@@@@@|...|
                d..|.......|@@@@@@@@@@@@@@@@@@@@|...|
                ...|.......|@@@@@@@@@@@@@@@@@@@@|...|..b.. and c is the depth in 3D
                ...v...|@@@@@@@@@@@@@@@@@@@@@@@@|...|
                .......|@@@@@@@@@@@@@@@@@@@@@@@@|...|
                ....................................v
                .......<----------------------->
                ...................a


    Attributes
    ----------
    size : ARRAY OF FLOAT or LIST
        defined from the size parameters
    type : STR
        shape of the domain containing the lattice
    """

    def __init__(self, size):
        """Create the domain."""
        if len(size) == 1:
            self.type = "ball"
            self.size = size
        elif len(size) == 2 or len(size) == 3:
            self.type = "rectangle"
            self.size = size
        elif len(size) == 5:
            self.type = "wedge"
            self.size = size
        else:
            raise ValueError("Illegal size. Retry with another input.")

    def __repr__(self):
        return "This domain is a {}-like box of dimensions {}.".\
            format(self.type, self.size)

#%% Definition of the general lattice object

class Lattice:
    """The lattice is the whole meshed network of nodes and elements.

    Lattices are essentially two tables: ``nodes`` which is the position in 
    space of the nodes, and ``elems`` which gives the link between nodes 
    based on their indices. The Lattice class allows easier manipulation 
    and characterization of this structure.

    Lattices are created either from already existing nodes and elements
    tables or initialized based on the wanted mesh with 
    :py:meth:`Lattice.initialize()`.

    Parameters
    ----------
    For the initialize method, see Notes below.
    nodes : numpy.ndarray
        Position of each node
    links : list or numpy.ndarrya
        if list: connection list of each node, if array: elements defined 
        by the two nodes connected
    size : numpy.ndarray, optional
        size of the treillis.Domain containing the lattice. Can be extracted 
        from the nodes.
    length : float, optional
        average length of the elements. Can be calculated from the nodes 
        and elements.
    inimesh : str, optional
        type of treillis.Mesh used to define the lattice
    aspect_ratio : float, optional, default is 10.
        aspect ratio of an element, defined as length/width
    beam_section : str or square numpy.ndarray, optional, default is 'Circular'
        shape of an element. Either 'Circular', 'Square', or a square 
        numpy.ndarray of 0s and 1s, where 1 indicates a position where 
        there is matter.
    fixed_nodes : numpy.ndarray, optional
        indices of the nodes defining the boundary conditions of the 
        lattice.


    Notes
    -----
    Lattices can be generated from scratch, using the
    :py:meth:`Lattice.initialize()` method. This method is based on different
    algorithms depending on whether the wanted lattice is periodical or not.
    For periodical lattices, a :py:class:`Mesh` instance is created, 
    representing the elementary cell, which is then repeated through rotations 
    and symmetries to fill the entire :py:class:`Domain`. For isotropic lattices,
    triangulation algorithms are performed on a random close packing of solid
    spheres.
    
        
    Attributes
    ----------
    nodes : (Nnodes, Ndim) numpy.ndarray
        Position of the nodes.
    elems : (Nelems, 3) numpy.ndarray
        Indices of the nodes forming the elements.
        
        .. deprecated:: 0.1
           Instead of the third column, prefer using the length property to 
           get the length of all elements.
           
    dim : int (2 or 3)
        Dimension of the lattice. 
    sizedom : numpy.ndarray
        Dimensions of the Domain.
    domain : Domain
        Domain containing the lattice.
    L : float
        Average length of the elements.
    type : str
        Type of mesh in the lattice.
    aspect_ratio : (Nelems,) numpy.ndarray
        Geometric ratio of each element expressed as length/width.
    node_section : float
        Average size of a node.
    beam_section : str or (Nelems, N, N) numpy.ndarray
        Shape of the beams.
    fixed_nodes : 1D numpy.ndarray
        Indices of the outer boundary nodes.
    length : (Nelems,) numpy.ndarray
        Length of the elements.
    angle : (Nelems, Ndim-1) numpy.ndarray
        2 or 3D angles of the elements.
    contacts_list : (Nnodes,) list
        List of indices of neighbor nodes for all nodes.
    coordination : (Nnodes,) numpy.ndarray
        Number of contact neighbor for each node.
    coord : float
        Average coordination.
    density : float
        Volume or surface fraction in the Domain occupied by the lattice.
    incidence : (Nnodes, Nelems) numpy.ndarray
        Table of incidence of the nodes in each element. 1 if node is the
        final node in the given element, -1 if node is the initial node, 0
        otherwise.
        
    Examples
    --------
    
    Generating a lattice from a random distribution of points.
    
    >>> import numpy as np
    >>> from scipy.spatial import Delaunay
    >>> import treillis
    >>> from treillis.display import clean as cd
    >>>
    >>> #%% Creation of the base for the lattice
    >>> # Generate points randomly distributed in a plane
    >>> # and link them together with a Delaunay triangulation
    >>> points = np.random.default_rng().uniform(size = [20, 2])
    >>> points -= points.mean(axis=0)
    >>> DT = Delaunay(points)
    >>> e1 = DT.simplices.ravel()
    >>> e2 = np.roll(DT.simplices, -1, axis=1).ravel()
    >>> links = np.concatenate(
    >>>    (e1.reshape(len(e1),1),
    >>>     e2.reshape(len(e2),1)),
    >>>    axis=1)
    >>>
    >>> #%% Creating the lattice
    >>> lat = treillis.Lattice(points, links)
    >>> lat
    2D None-type lattice with 20 nodes and 93 elements in a rectangle-shaped Domain.
    >>> cd.showlat(lat, axis=False, box=False)
    """

    def __init__(self, nodes, links, *,
                 size=None, length=None, inimesh=None,  aspect_ratio=10.,
                 beam_section="Circular", fixed_nodes=None):
        if aspect_ratio is None: aspect_ratio = 10.
        
        self.nodes: np.ndarray = nodes
        
        self.dim = nodes.shape[1]

        # Find if links are elements or a contact list per node
        if isinstance(links, np.ndarray)\
            or len(np.unique([len(l) for l in links]))==1:
                elems = True
        else: elems = False

        # Construct the element table
        if not elems:
            elems = []
            for i in range(len(links)):
                for j in links[i]:
                    if [j,i] not in elems:
                        elems.append([i,j])

            self.elems = np.array(elems)
        else:
            if isinstance(links, np.ndarray): self.elems = links
            else: self.elems = np.array(links)

        # Calculate element length
        lenelem = np.sqrt(
            np.sum( (self.nodes[self.elems[:,0].astype(int)]
                     -self.nodes[self.elems[:,1].astype(int)])**2, axis=1 ))
        self.elems = np.concatenate( (self.elems[:,:2],
                                      lenelem.reshape(len(self.elems),1)),
                                    axis=1)

        # Define the other attributes
        if size is None:
            self.sizedom = np.ptp(nodes, axis=0)
        else: self.sizedom = size

        self.domain = Domain(self.sizedom).type

        if length is None: self.L = np.mean(lenelem)
        else: self.L = length

        self.type = inimesh
        if isinstance(aspect_ratio, np.ndarray)\
            and len(aspect_ratio) != len(self.elems):
            print("Warning: length discrepancy with the aspact_ratio, only average is kept.")
            aspect_ratio = np.mean(aspect_ratio)
        if not isinstance(aspect_ratio, np.ndarray):
            self.aspect_ratio = np.ones((len(self.elems),)) * aspect_ratio
        else: self.aspect_ratio = aspect_ratio
        self.node_section = self.L/aspect_ratio
        self.beam_section = beam_section
        if isinstance(beam_section, np.ndarray):
            if len(beam_section.shape)==2:
                self.beam_section = np.repeat(
                    beam_section.reshape((1,beam_section.shape[0],
                                          beam_section.shape[1])),
                    len(self.elems), axis=0)
            if len(beam_section.shape)==3:
                if beam_section.shape[2]!=len(self.elems):
                    self.beam_section = 'Circular'
                    print("Warning: section not well defined, falling back to 'Circular' for init.")

        if fixed_nodes is not None:
            self._fixed_nodes = np.array(fixed_nodes)
        else:
            self._fixed_nodes = None
        self._density = None

    @classmethod
    def initialize(cls, size, length, inimesh, *, aspect_ratio=10.,
                 beam_section="Circular", prepacking = True, easy = True,
                 starting=None, des=0):
        """
        Create a lattice from a base mesh.

        With the initialize method, a lattice is created from scratch, by
        creating an initial base cell from the mesh, which is repeated
        periodically within the domain with rotations and symmetries to fill
        the space occupied by the lattice.

        Parameters
        ----------
        size : numpy.ndarray
            size of the treillis.Domain containing the lattice
        length : float
            average length of an element
        inimesh : str
            type of treillis.Mesh used to construct the lattice
        aspect_ratio : float, optional, default is 10.
            average aspect ratio of an element
        beam_section : str or numpy.ndarray, optional, default is 'Circular'
            shape of an element section
        prepacking : bool, optional, default is False
            use an elready existing packing to shorten the calculation time 
            for Voronoi- and Delaunay-based lattices
        easy : bool, optional, default is True
            Slightly modified Voronoi with fewer, and more monodisperse 
            elements
        starting : None or str, optional, default is None
            type of lattice to use as a base for the Voronoi lattice 
            instead of a random close packing
        des : float, optional, default is 0
            disorder parameter, adds some variation in beam length at 
            contruction for Voronoi- and Delaunay-based lattices. For 2-D
            lattices to be isotropic, des needs to be >0. For 3D Voronoi
            with a starting lattice, it is the shake parameter.

        Yields
        ------
        treillis.Lattice
        
        Examples
        --------
        Generate a wedge-splitting configuration Delaunay-base lattice.
        
        >>> import numpy as np
        >>> import treillis
        >>> lat = treillis.Lattice.initialize(
        >>>     np.array([120, 100, 35, 40, 30]), 6, 'TriDT3D')
        >>> lat
        3D TriDT3D-type lattice with 2140 nodes and 12465 elements in a wedge-shaped Domain.

        The domain type is guessed from the length of the size array, and 
        the mesh from the inimesh.

        See Also
        --------
        treillis.Mesh
        treillis.Domain
        
        References
        ----------
        
        - Lozano et al., 2016 [https://doi.org/10.1016/j.camwa.2016.02.032]
        - Baranau and Tallarek, 2021 [https://doi.org/10.1063/5.0036411]
        
        
        """
        if inimesh not in ["Hexa", "Square", "Tri", "TriDT", "Cubic", "TriDT3D",
                           "TriPrism", "HexaPrism", "Octa", "Dodec", "Octet",
                           "Vor", "Vor3D"]:
            raise AttributeError("Unknown starting mesh.")

        def AddToLat(nodes, mesh, domain, bar, compteur):
            """
            Check if new mesh is included in the domain.

            Method that verifies if each point of new mesh is included in the
            domain and if new mesh is not already (using barycentres comparison).

            Parameters
            ----------
            compteur : INT
                variable used in the iterative process to keep a record of the
                number of meshes copied.

            Returns
            -------
            compteur : INT
                variable used in the iterative process to keep a record of the
                number of meshes copied.

            """
            EPS = 1E-4
            table_dist_bar = np.absolute(bar[:-1, :] - \
                            bar[-1, :].reshape(1, mesh.dim)) < EPS
            nodes2test = nodes[-mesh.size:, :]

            if domain.type == "ball":
                domain_test = np.prod(np.sqrt(np.sum(nodes2test ** 2, axis=1)) <\
                                      domain.size[0] + EPS)

            if domain.type == 'rectangle':
                domain_test = np.prod((np.abs(nodes2test[:, 0]) <\
                           domain.size[0] / 2 + EPS) * (np.abs(nodes2test[:, 1]) \
                                                     < domain.size[1] / 2 + EPS))
                if mesh.dim == 3:
                    domain_test = domain_test * np.prod(np.abs(nodes2test[:, 2]) <\
                                                        domain.size[2] / 2 + EPS)

            if domain.type == "wedge":
                a = domain.size[0]
                b = domain.size[1]
                d = domain.size[3]
                e = domain.size[4]

                domain_subtestright = np.prod(nodes2test[:, 1] > -b / 2 - EPS) and \
                                      np.prod(nodes2test[:, 1] < b / 2 + EPS) and \
                                np.prod(nodes2test[:, 0] > (e - a / 2) + EPS) and \
                                      np.prod(nodes2test[:, 0] < a / 2 + EPS)

                domain_test = np.prod(nodes2test[:, 0] > -a / 2 - EPS) and \
                              np.any(nodes2test[:, 0] < (e - a / 2) + EPS) and \
                              np.prod(nodes2test[:, 1] > -b / 2 - EPS) and \
                              np.prod(nodes2test[:, 1] < b / 2 + EPS) and \
                              (np.prod(nodes2test[:, 1] < -d / 2 + EPS) or \
                              np.prod(nodes2test[:, 1] > d / 2 - EPS))

                domain_test = domain_test or domain_subtestright

                if mesh.dim == 3:
                    c = domain.size[2]
                    domain_test = domain_test *\
                        np.prod(np.abs(nodes2test[:, 2]) < c / 2 + EPS)

            if domain_test and not(np.sum(np.prod(table_dist_bar, axis=1), axis=0)):
                compteur += 1

            else:
                compteur += 0
                nodes = np.delete(nodes, np.s_[-mesh.size:], axis=0)
                bar = np.delete(bar, -1, axis=0)

            return compteur, bar, nodes


        def Sym(mesh, nodes, ref):
            """
            Create axi-symmetric nodes.

            The 2D coordinates of the symmetric of nodes are found with respect to
            a line defined by ref

            Parameters
            ----------
            nodes: (1,N) ARRAY OF FLOAT
                node whose symmetric coordinates you want.

            ref: (2,N) or (3,N) ARRAY OF FLOAT
                array of the two nodes that define the line.

            Returns
            -------
            ARRAY OF FLOAT
                coordinates of the symmetric of nodes.

            """
            if mesh.dim == 2:
                return Sym2D(ref[0, :], ref[1, :], nodes)

            return Sym3D(ref[0, :], ref[1, :], ref[2, :], nodes)


        if not isinstance(size, np.ndarray):
            if isinstance(size, list):
                size = np.array(size)
            else:
                size=np.array([size])

        # Mesh initialization
        mesh = call(length, inimesh, starting)
        if mesh.dim==3 and len(size) in [2]:
            raise AttributeError("Create a 3D box for this 3D mesh.")

        if mesh.dim==2 and len(size) in [3,5]:
            print("The out of plane direction will not be taken into account for this 2D mesh.")
            if len(size)==3:
                size=size[:2]

        if mesh.dim == 2:
            sides = mesh.sides
        else:
            sides = mesh.faces


        nodes = mesh.point
        bar = np.reshape(mesh.bar, (1, mesh.dim))
        elems = np.zeros((0, 3), dtype=TYPEFLOAT)
        faceslist = mesh.faceslist

        # Domain creation
        domain = Domain(size)


        # Build the periodical lattice iteratively
        if mesh.size != 0:
            # We define the lattice as a set of mesh points contained in an array
            # and incremented when calling fonction AddToLat
            compteur = 0
            count = [0, 1]
            count_cum = np.cumsum(count)
            t = 1

            while count[t] != 0:
                count += [0]

                # We scan all the meshes created at iteration t:
                for h in range(count[t]):
                    ref_mesh = nodes[(h + count_cum[t - 1]) *\
                          mesh.size:(h + 1 +  count_cum[t - 1]) * mesh.size, :]

                    # We create as many new meshes as the connectivity:
                    if type(sides) is list:
                        sides1 = sides[0]
                    else:
                        sides1 = sides

                    for i in range(sides1):
                        if type(faceslist) is list:
                            nodes_ref = faceslist[0][i, :]
                        else:
                            nodes_ref = faceslist[i, :]

                        nodes_ref = nodes_ref[~np.isnan(nodes_ref)]
                        nodes_ref = nodes_ref.astype(int)
                        nodes2symmetrize = np.arange(0, mesh.size, dtype=int)
                        nodes2symmetrize = np.setdiff1d(nodes2symmetrize,
                                                        nodes_ref)
                        ref_sym = ref_mesh[nodes_ref, :]

                        nodes = np.concatenate((nodes,
                                     np.zeros((mesh.size, mesh.dim))), axis=0)

                        nodes[-mesh.size + nodes_ref, :] = ref_sym

                        # We construct the new mesh's nodes using symmetry properties:
                        for j in range(mesh.size - len(nodes_ref)):
                            nodes[-mesh.size + nodes2symmetrize[j], :] = \
                            Sym(mesh, ref_mesh[nodes2symmetrize[j], :], ref_sym)

                        if mesh.size == 24 and len(nodes_ref) == 6:
                            nodes[-mesh.size:, :] = Rot3D(ref_sym,
                                      nodes[-mesh.size:, :])

                        bar_new = np.sum(nodes[-mesh.size:, :], axis=0) /\
                            mesh.size
                        bar_new = np.reshape(bar_new, (1, mesh.dim))
                        bar = np.concatenate((bar, bar_new), axis=0)

                        # We add the mesh created to lat list under some conditions:
                        compteur, bar, nodes = AddToLat(nodes, mesh, domain, bar, compteur)

                t += 1
                count[t] = compteur
                count_cum = np.cumsum(count)
                compteur = 0

            # We define the elems constituting the lattice giving the indexes of
            # their incoming/outgoing nodes in the nodes array and their length:
            arr_in = np.zeros((0, mesh.dim), dtype=TYPEFLOAT)
            arr_out = np.zeros((0, mesh.dim), dtype=TYPEFLOAT)

            for i in range(np.shape(bar)[0]):
                if type(faceslist) is list:
                    nodes_in = (faceslist[1] + mesh.size * i).ravel()
                else:
                    nodes_in = (faceslist + mesh.size * i).ravel()

                nodes_in = nodes_in[~np.isnan(nodes_in)]
                nodes_in = nodes_in.astype(int)
                arr_in = np.concatenate((arr_in, nodes[nodes_in, :]), axis=0)

                if type(sides) is list:
                    sides2 = sides[1]
                else:
                    sides2 = sides

                for j in range(sides2):
                    if type(faceslist) is list:
                        nodes_out = (faceslist[1] + mesh.size * i)[j, :]
                    else:
                        nodes_out = (faceslist + mesh.size * i)[j, :]

                    nodes_out = nodes_out[~np.isnan(nodes_out)]
                    nodes_out = np.roll(nodes_out, -1)
                    nodes_out = nodes_out.astype(int)
                    arr_out = np.concatenate((arr_out, nodes[nodes_out, :]),
                                             axis=0)

            arr_mid = (1 / mesh.dim) * (arr_in + arr_out)
            arr_mid = np.around(arr_mid, decimals=DECIMAL)

            arr_in = np.around(arr_in, decimals=DECIMAL)
            arr_out = np.around(arr_out, decimals=DECIMAL)

            index2 = np.unique(arr_mid, axis=0, return_index=True)[1]
            arr_in = arr_in[np.sort(index2)]
            arr_out = arr_out[np.sort(index2)]

            # We define the nodes constituting the lattice giving their
            # cartesian coordinates:
            nodes = np.around(nodes, decimals=DECIMAL)
            index1 = np.unique(nodes, axis=0, return_index=True)[1]
            nodes = nodes[np.sort(index1)]

            # We identify the elems giving the indexes of node_in and node_out
            ## in nodes array:
            for i in range(len(index2)):
                node_in = np.nonzero(np.prod(nodes == arr_in[i, :],
                          axis=1))[0].item()
                node_out = np.nonzero(np.prod(nodes == arr_out[i, :],
                          axis=1))[0].item()
                #array = np.array([[node_in, node_out, 0.]], dtype = TYPEFLOAT)
                listed_array = [[node_in, node_out, 0.]]
                array = np.array(listed_array, dtype = TYPEFLOAT)
                elems = np.concatenate((elems, array), axis=0)

            elems[:,2] = np.sqrt(np.sum((nodes[elems[:,1].astype(int)] -\
                nodes[elems[:,0].astype(int)])**2, axis=1))

            fixed_nodes = None

        else:
            nodes = np.empty((0,mesh.dim))
            elems = np.empty((0,3))
            if (mesh.type in ['Vor', 'Vor3D'] and starting is None)\
                or (mesh.type in ['TriDT', 'TriDT3D']):
                starting = makepacking(mesh.dim, domain.size, 
                                       length\
                                           * (1+.33*(mesh.type=='Vor3D'))\
                                           / (1+.33*(mesh.type=='Vor')), 
                                           des, prepacking)\
                    * (('Vor' in mesh.type)*np.sqrt(3) + ('TriDT' in mesh.type))
                # this last line allows to have the correct length of elements
            elif mesh.type in ['Vor', 'Vor3D'] and starting is not None:
                mesh_start = call(length, starting)
                print("Generating starting lattice")
                starting = Lattice.initialize(
                    domain.size, length, starting)
                starting = shake(starting, des)
                starting = starting.nodes
                if 'Vor' in mesh.type:
                    if 'Tri' in mesh_start.type: starting *= np.sqrt(3)
                    if 'Hexa' in mesh_start.type: starting /= np.sqrt(3)
                    elif mesh.dim==3: print('The length of the elements may not be correct, you can just multiply the nodes table and ecalculate them to get them right.')

            print("Creating lattice")
            nnodes, nelems, fixed = frompacking(starting, mesh.type,
                                                domain.size, length, mesh.dim,
                                                easy=easy)

            nodes = np.concatenate((nodes, nnodes), axis=0)
            elems = np.concatenate((elems, nelems), axis=0)
            fixed_nodes = fixed.tolist()

        return cls(nodes, elems, size=size, length=length, inimesh=mesh.type,
                   aspect_ratio=aspect_ratio, beam_section=beam_section,
                   fixed_nodes=fixed_nodes)


    @property
    def length(self):
        """Get the length of all elements."""
        L = np.sqrt(np.sum(
            (self.nodes[self.elems[:,1].astype(int)]
             - self.nodes[self.elems[:,0].astype(int)])**2
            , axis=1))
        if self.elems.shape[1]==3: self.elems[:,2] = L
        return L

    @property
    def angle(self):
        """Return the polar angle in 2D, and the polar and azimuthal angles of the elems in 3D."""
        vec = (self.nodes[self.elems[:,1].astype(int)]
         - self.nodes[self.elems[:,0].astype(int)])\
            / np.repeat(self.length.reshape(self.elems.shape[0],1),
                        self.dim, axis=1)
        theta = np.atan2(vec[:,1], vec[:,0])
        if self.dim == 2:
            return theta
        else:
            phi = np.arccos(vec[:,2] / np.sqrt(np.sum(vec**2, axis=1)))
            angle = np.concatenate((
                phi.reshape(self.elems.shape[0],1),
                theta.reshape(self.elems.shape[0],1)),
                axis=1)
            return angle
        
    def anisotropy(self):
        """Calculate a structural anisotropy index as a distance to a material with fully random angles."""
        pbins = np.arange(-180,180)*np.pi/180
        tbins = np.arange(180)*np.pi/180
        if self.dim==2:
            h = np.histogram(self.angle, pbins, density=True)
            return np.sqrt(
                sum((h[0]-h[0].mean())**2))
        else:
            ph = np.histogram(self.angle[:,1], pbins, density=True)[0]
            th = np.histogram(self.angle[:,0], tbins, density=True)[0]
            x = (tbins[1:]+tbins[:-1])/2
            return np.sqrt(
                sum((ph-ph.mean())**2) + sum((th-np.sin(x))**2)) 
        
    @property
    def contacts_list(self):
        """Get the list of neighboring nodes for each node."""
        nodes = self.nodes
        elems = self.elems[:,:2].astype(int)
        
        coord = [np.nonzero(elems==i) for i in range(nodes.shape[0])]
        contacts_save = [elems[c[0], 1-c[1]].astype(int) for c in coord]
        return contacts_save

    @property
    def coordination(self):
        """Connectivity per node."""
        @jit(parallel=True)
        def Z(nodes, elems):
            """Get the list of neighboring nodes for each node."""
            z = [len(np.flatnonzero(elems[:,:2]==i)) 
             for i in prange(nodes.shape[0])]
            return np.array(z)
        
        z = Z(self.nodes, self.elems[:,:2].astype(int))
        return z.reshape((len(z),1))

    @property
    def coord(self):
        """Average connectivity of the lattice."""
        return np.mean(self.coordination)

    @property
    def density(self):
        """Measure the density (as the volume fraction or surface fraction of solid) of the lattice."""
        if self._density is None:
            try:
                d = Density2(self)
            except:
                print("Warning: the density calculated exactly with Voronoi failed. Density is approximated from the box.")
                d = Density(self)
            self._density = d
            return d
        else: return self._density

    @property
    def fixed_nodes(self):
        """Nodes on the external boundaries of the lattice."""
        if hasattr(self, '_fixed_nodes'):
            if self._fixed_nodes is not None:
                if self._fixed_nodes.shape[0] == 0: 
                    self._fixed_nodes = None
                else:
                    return self._fixed_nodes

        fixed = []
        fixed_final = []
        
        try: self.nodes[self._fixed_nodes]
        except: 
            self._fixed_nodes = None
            print("There is an issue with your fixed nodes choice, determining them with the default method.")

        if self._fixed_nodes is not None:
            return self._fixed_nodes

        elif self.type in ['Hexa', 'Square', 'Tri', 'Cubic',
                           'TriPrism', 'HexaPrism', 'Octa',
                           'Dodec', 'Octet']:
            fixed = np.nonzero(np.squeeze(self.coordination <= \
                        np.floor(self.coord)-1))[0].tolist()

            fixed_final = np.unique(fixed).astype(int)

        else: # Adding "artificial" nodes around, to see who links to them
            print('Calculating the fixed nodes')
            nodes = self.nodes
            Nnodes = len(nodes)
            if self.dim==2:
                # rectangular domain -
                if self.domain in ['rectangle', 'wedge']:
                    x0 = min(nodes[:,0])-self.L/3
                    x1 = max(nodes[:,0])+self.L/3
                    y0 = min(nodes[:,1])-self.L/3
                    y1 = max(nodes[:,1])+self.L/3
                    Nx = int((x1-x0)/self.L*2)+1
                    Ny = int((y1-y0)/self.L*2)+1

                    left = np.concatenate((
                        np.array([x0]*Ny).reshape(Ny,1),
                        np.arange(y0, y1, self.L/2).reshape(Ny,1)),
                        axis=1)

                    right = np.concatenate((
                        np.array([x1]*Ny).reshape(Ny,1),
                        np.arange(y0, y1, self.L/2).reshape(Ny,1)),
                        axis=1)

                    up = np.concatenate((
                        np.arange(x0, x1, self.L/2).reshape(Nx,1),
                        np.array([y1]*Nx).reshape(Nx,1)),
                        axis=1)

                    down = np.concatenate((
                        np.arange(x0, x1, self.L/2).reshape(Nx,1),
                        np.array([y0]*Nx).reshape(Nx,1)),
                        axis=1)

                    addnode = np.concatenate(
                        (left, right, up, down),
                        axis=0)
                # spherical domain
                else:
                    R = np.mean([max(nodes[:,0]), max(nodes[:,1])])+self.L/3
                    N = int(2*np.pi*R/self.L*2)+1
                    theta = np.linspace(0, 2*np.pi, N)
                    addnode = np.concatenate(
                        ((R*np.cos(theta)).reshape(N,1),
                         (R*np.sin(theta)).reshape(N,1)),
                        axis=1)

            else:
                # rectangular domain -
                if self.domain in ['rectangle', 'wedge']:
                    x0 = min(nodes[:,0])-self.L/3
                    x1 = max(nodes[:,0])+self.L/3
                    y0 = min(nodes[:,1])-self.L/3
                    y1 = max(nodes[:,1])+self.L/3
                    z0 = min(nodes[:,2])-self.L/3
                    z1 = max(nodes[:,2])+self.L/3
                    Nx = int((x1-x0)/self.L*2)+1
                    Ny = int((y1-y0)/self.L*2)+1
                    Nz = int((z1-z0)/self.L*2)+1

                    XY = np.meshgrid(
                        np.linspace(x0, x1, Nx),
                        np.linspace(y0, y1, Ny)
                        )
                    YZ = np.meshgrid(
                        np.linspace(y0, y1, Ny),
                        np.linspace(z0, z1, Nz)
                        )
                    XZ = np.meshgrid(
                        np.linspace(x0, x1, Nx),
                        np.linspace(z0, z1, Nz)
                        )

                    down = np.concatenate(
                        (XY[0].ravel().reshape(Nx*Ny,1),
                         XY[1].ravel().reshape(Nx*Ny,1),
                         np.array([z0]*Nx*Ny).reshape(Nx*Ny,1)),
                        axis=1)

                    up = np.concatenate(
                        (XY[0].ravel().reshape(Nx*Ny,1),
                         XY[1].ravel().reshape(Nx*Ny,1),
                         np.array([z1]*Nx*Ny).reshape(Nx*Ny,1)),
                        axis=1)

                    left = np.concatenate(
                        (np.array([x0]*Ny*Nz).reshape(Ny*Nz,1),
                         YZ[0].ravel().reshape(Ny*Nz,1),
                         YZ[1].ravel().reshape(Ny*Nz,1)),
                        axis=1)

                    right = np.concatenate(
                        (np.array([x1]*Ny*Nz).reshape(Ny*Nz,1),
                         YZ[0].ravel().reshape(Ny*Nz,1),
                         YZ[1].ravel().reshape(Ny*Nz,1)),
                        axis=1)

                    front = np.concatenate(
                        (XZ[0].ravel().reshape(Nx*Nz,1),
                         np.array([y0]*Nx*Nz).reshape(Nx*Nz,1),
                         XZ[1].ravel().reshape(Nx*Nz,1)),
                        axis=1)

                    back = np.concatenate(
                        (XZ[0].ravel().reshape(Nx*Nz,1),
                         np.array([y1]*Nx*Nz).reshape(Nx*Nz,1),
                         XZ[1].ravel().reshape(Nx*Nz,1)),
                        axis=1)

                    addnode = np.concatenate(
                        (up, down, left, right, front, back),
                        axis=0)

                # spherical domain
                else:
                    R = np.max(
                        np.sqrt(np.sum(self.nodes**2, axis=1)))+self.L/3
                    Np = int(2*np.pi*R/self.L*2)+1
                    Nt = int(np.pi*R/self.L*2)+1
                    phi = np.linspace(0, 2*np.pi, Np)
                    theta = np.linspace(0, np.pi, Nt)

                    ANGLES = np.meshgrid(theta, phi)
                    addnode = np.concatenate(
                        ((R * np.sin(ANGLES[0].ravel())\
                          * np.cos(ANGLES[1].ravel())).reshape(Np*Nt,1),
                         (R * np.sin(ANGLES[0].ravel())\
                           * np.sin(ANGLES[1].ravel())).reshape(Np*Nt,1),
                         (R * np.cos(ANGLES[0])).reshape(Np*Nt,1)),
                        axis=1)

            nodes = np.concatenate((nodes, addnode), axis=0)
            DT = Delaunay(nodes)
            elem1 = DT.simplices.ravel()
            elem2 = np.roll(DT.simplices, -1, axis=1).ravel()
            elemsDT = np.concatenate((elem1.reshape((len(elem1), 1)),
                                      elem2.reshape((len(elem2), 1))), axis=1)
            elemsDT.sort(axis=1)
            elemsDT = np.unique(elemsDT, axis=0).astype(float)
            test = (elemsDT[:,0] >= Nnodes) | (elemsDT[:,1] >= Nnodes)
            
            if self.dim == 2:
                who = np.nonzero(elemsDT < Nnodes)
                test = (elemsDT[who[0],0] >= Nnodes)\
                    | (elemsDT[who[0],1] >= Nnodes)
                o = np.unique(elemsDT[who[0][test],who[1][test]])
                old = self.elems[np.prod(np.isin(self.elems[:,:2], o), 
                                         axis=1, dtype=bool),:2]
                new = elemsDT[np.prod(np.isin(elemsDT[:,:2], o), 
                                         axis=1, dtype=bool),:2]
                
                diff = np.prod(np.isin(old, new, invert=True), axis=1, 
                               dtype=bool)
                diff = old[diff]
                coord = [np.nonzero(new == d) for d in diff[:,0]]
                ct1 = [new[c[0], 1-c[1]] for c in coord]
                coord = [np.nonzero(new == d) for d in diff[:,1]]
                ct2 = [new[c[0], 1-c[1]] for c in coord]
                not_lim = np.array(
                    [c1[np.isin(c1, c2)] for (c1, c2) in zip(ct1, ct2)])
                
                fixed_final = o[np.isin(o, not_lim, invert=True)]
            
            else:
                for elm in elemsDT[test][:,:2]:
                    if sum(elm<Nnodes)>0: fixed_final.append(elm[elm<Nnodes])

            fixed_final = np.unique(fixed_final).astype(int)
            
            # if self.type in ['TriDT3D', 'Vor3D']:
            #     fixed_final = fixed_final[
            #         (self.coordination[fixed_final] <= 12*(self.type=='TriDT3D')
            #             + 3*(self.type=='Vor3D')).flatten()
            #             ]
        self._fixed_nodes = fixed_final
        return fixed_final

    @property
    def incidence(self):
        """Table indicating which node is implicated in which link with orientation, for tensorial calculation."""
        incid = np.zeros((len(self.nodes), len(self.elems)))

        incid[self.elems[:,0],np.arange(len(self.elems))] = -1
        incid[self.elems[:,1],np.arange(len(self.elems))] = 1
        return incid

    def del_elm(self, indx_elm):
        """Delete the elems at indx_elm."""
        try: len(indx_elm)
        except: indx_elm = np.array([indx_elm])
        if not isinstance(indx_elm, np.ndarray): indx_elm = np.array(indx_elm)
                                                
        self.elems = np.delete(self.elems, indx_elm, axis=0)
        self.aspect_ratio = np.delete(self.aspect_ratio, indx_elm, axis=0)
        if isinstance(self.beam_section, np.ndarray):
            self.beam_section = np.delete(self.beam_section, indx_elm, axis=0)
        self._density = None
        pass
    
    def del_node(self, indx_node):
        """Delete the nodes at indx_node, and all the elements linked to these nodes."""
        I_keep = np.arange(self.nodes.shape[0])
        I_keep = np.delete(I_keep, indx_node)
        self.nodes = np.delete(self.nodes, indx_node, axis=0)
        I_elm = np.sum(np.isin(self.elems[:,:2], indx_node), axis=1, dtype=bool)
        self.del_elm(np.flatnonzero(I_elm))
        if self._fixed_nodes is not None:
            self._fixed_nodes = np.delete(self._fixed_nodes,
                                          np.isin(self._fixed_nodes, indx_node))
        
        for i in range(len(I_keep)):
            self.elems[self.elems[:,0]==I_keep[i],0] = i
            self.elems[self.elems[:,1]==I_keep[i],1] = i
            if self._fixed_nodes is not None:
                if self._fixed_nodes.shape[0] == 0:
                    self._fixed_nodes = None
                else:
                    self._fixed_nodes[self.fixed_nodes==I_keep[i]] = i
        self._density = None
        pass

    def clean_lattice(self):
        """Eliminate nodes that are unlinked or linked to only one neighbor."""
        @jit(parallel=True)
        def Z(nodes, elems):
            """Get the list of neighboring nodes for each node."""
            z = [len(np.flatnonzero(elems[:,:2]==i)) 
             for i in prange(nodes.shape[0])]
            return np.array(z)

        def length(nodes, elems):
            return np.sqrt(np.sum(
                (nodes[elems[:,0].astype(int),:]\
                 -nodes[elems[:,1].astype(int),:])**2, axis=1))

        ct = Z(self.nodes, self.elems[:,:2].astype(int))
        l = length(self.nodes, self.elems)
        nodes = deepcopy(self.nodes)
        elems = deepcopy(self.elems)
        ratio = deepcopy(self.aspect_ratio)
        fixed = self.fixed_nodes
        if isinstance(self.beam_section, np.ndarray):
            section = deepcopy(self.beam_section)
        else: section = None
        
        itera = 0

        while 0 in l or (ct<=1).any():
            print('Iteration #', itera+1)
            itera += 1
            # Remove 0-length elements
            I0 = np.arange(len(elems))[l==0]
            I_fuse = elems[I0,:2]
            elems = np.delete(elems, I0, axis=0)
            ratio = np.delete(ratio, I0)
            if section is not None: section = np.delete(section, I0)

            # Fuse nodes that are superimposed by deleting in elems
            for I in I_fuse:
                elems[elems[:,0]==I[1],0] = I[0]
                elems[elems[:,1]==I[1],1] = I[0]

            # Delete isolated nodes
            ct = Z(nodes, elems[:,:2].astype(int))
            I_keep = np.arange(len(nodes))[ct>1]
            I_n = np.arange(nodes.shape[0])[I_keep]
            fixed = np.flatnonzero(np.isin(I_n,fixed))
            nodes = nodes[I_keep,:]

            keep = np.array([np.prod(np.isin(e, I_keep)) for e in elems[:,:2]],
                            dtype=bool)
            
            elems = elems[keep]
            ratio = ratio[keep]
            if section is not None: section = section[bool(keep)]

            for i in range(len(I_keep)):
                elems[elems[:,0]==I_keep[i],0] = i
                elems[elems[:,1]==I_keep[i],1] = i

            ct = Z(nodes, elems)
            l = length(nodes, elems)

        self.nodes = nodes
        self.elems = elems
        self.aspect_ratio = ratio
        self._fixed_nodes = fixed
        self._density = None
        if section is not None: self.beam_section = section
        
    def __repr__(self):
        return str(self.dim)+'D '+str(self.type)+'-type lattice with '+\
            str(self.nodes.shape[0])+' nodes and '+str(self.elems.shape[0])\
                +' elements in a '+self.domain+'-shaped Domain.'


def shake(lattice, delta, in_place=True):
    """
    Shake the lattice nodes to give an amorphous feeling.

    Method that deforms the lattice in order to amorphize it.
    The points are moved randomly with a uniform picked theta rotation
    and a translation defined by delta.

    Parameters
    ----------
    lattice: treillis.Lattice
        base lattice to shake
    delta : float
        disorder intensity.
    in_place: bool
        Whether to change the lattice in place or create a new lattice, default is True.

    Raises
    ------
    ValueError
        disorder intensity must be >=0 and <0.5.

    Returns
    -------
    shaken: treillis.Lattice
        shaken lattice
        
    Examples
    --------
    >>> # Load a periodical lattice
    >>> import treillis
    >>> lat = treillis.loadLattice('lattice_131')
    >>> # Shake it to add some isotropy
    >>> shaken = treillis.shake(lat, .25, False)
    >>> shaken.anisotropy(), lat.anisotropy()
    (np.float64(5.519077388441031), np.float64(35.25821065856917))
    
    Anisotropy has been reduced by a factor 7.

    """
    if not in_place: lattice = deepcopy(lattice)
    nodes_init = deepcopy(lattice.nodes)
    contacts = lattice.contacts_list

    if delta > .5:
        delta = .5
        print("Warning: Disorder parameter value capped at 0.5")

    for i in range(len(lattice.nodes)):
        # Get the radius of the sphere within which the nodes can move
        r_neighbors = np.sqrt(np.sum(
            (nodes_init[i]-nodes_init[contacts[i]])**2, axis=1))
        r_sphere=np.min(r_neighbors)

        # Application of the disorder parameter
        rng = np.random.default_rng()
        S = rng.uniform(-1, 1)
        r = (delta * r_sphere) * np.sqrt(1 - S**2)

        # create a random angle
        if lattice.dim==2:
            theta = rng.uniform(0, 2*np.pi)
            lattice.nodes[i] += [r*np.cos(theta), r*np.sin(theta)]
        else:
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2*np.pi)
            lattice.nodes[i] += [r*np.sin(theta)*np.cos(phi),
                                 r*np.sin(theta)*np.sin(phi),
                                 r*np.cos(theta)]

    return lattice

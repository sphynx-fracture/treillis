"""
Meshes
======

Base mesh creation for periodical lattices.

Repository for 5 2D initial meshes and 8 3D initial meshes under the form
of classes. When using this module, you can just create the object mesh and
change the side's length but not the position in space which is always around
(0, 0) in 2D and (0, 0, 0) in 3D.
"""

#%% Importation of useful packages


import numpy as np
TYPEFLOAT = np.float64

__all__ = ['call', 'Mesh']


#%% Shortcuts for fast mesh import

def call(length, name, startinglattice = None):
    """
    Simplify the creation and use of the initial mesh.

    Use this function to generate your mesh before passing it as an argument
    for lattice class. Use this function instead of generating the mesh directly.

    Parameters
    ----------
    length : FLOAT
             corresponding to the element's length
    name : STRING
           corresponding to the name of the mesh you want for your lattice
           possible choices are:
               
           - Hexa
           - Square
           - Tri
           - TriDT
           - Cubic
           - TriDT3D
           - TriPrism
           - HexaPrism
           - Octa
           - Dodec
           - Octet
           - Vor (for this one is is possible to choose a starter lattice)
           - Vor3D (for this one is is possible to choose a starter lattice)
           
    startinglattice : STRING, optional
                       corresponding to the name of the starter lattice
                        you want to used as a base for Voronoi tesselation.

    Returns
    -------
    mesh : MESH OBJECT
           an instance of Mesh that will be used to create the lattice

    """
    if name == "Hexa":
        mesh = HexaMesh2D(length)
    elif name == "Square":
        mesh = SquareMesh2D(length)
    elif name == "Tri":
        mesh = TriMesh2D(length)
    elif name == "TriDT":
        mesh = TriDTMesh2D(length)
    elif name == "Cubic":
        mesh = CubicMesh3D(length)
    elif name == "TriDT3D":
        mesh = TriDTMesh3D(length)
    elif name == "TriPrism":
        mesh = TriMesh3D(length)
    elif name == "HexaPrism":
        mesh = HexaMesh3D(length)
    elif name == "Octa":
        mesh = TruncatedOctahedron3D(length)
    elif name == "Dodec":
        mesh = RhombicDodecahedron3D(length)
    elif name == "Octet":
        mesh = OctetTruss3D(length)
    elif name == "Vor" and startinglattice is not None:
        mesh = VoronoiMesh2D(length, startinglattice)
    elif name=="Vor3D" and startinglattice is not None:
        mesh = VoronoiMesh3D(length, startinglattice)
    elif name == "Vor" and startinglattice is None:
        mesh = VoronoiMesh2D(length)
    elif name=="Vor3D" and startinglattice is None:
        mesh = VoronoiMesh3D(length)
    else:
        mesh = Mesh(length)

    return mesh


#%% Definition of the general mesh object


class Mesh:
    """General object to describe the mesh.

    The mesh describes the elementary cell to build the lattice. The base cell
    can be:
        
    - an equilateral triangle (2D or 3D)
    - a Delauney-triangulation-base matrix (2D or 3D)
    - a Voronoi cell (2D or 3D)
    - a square cell (2D or 3D --cubic)
    - an hexahedric cell (2D or 3D)
    - a rhombic dodecahedric cell (3D)
    - a truncated octahedron (3D)

    Attributes
    ----------
    L : FLOAT
        side length of the base cell
    bar : ARRAY OF FLOATS
        orientation of the mesh
    coord : INT
        coordination number: number of closest neighbors
    dim : INT (2 or 3)
        dimensionality of the mesh
    faces : INT (None in 2D)
        number of faces in 3D
    faceslist : LIST OF ARRAY OF INTS
        list of the points joined by the face at position i
    point : (size,2) or (size,3) ARRAY OF INTS
        position of each point of the mesh
    sides : INT
        number of sides
    size : INT
        number of points
    type : STR
        type of the chosen mesh
    """

    def __init__(self, L):
        """

        Build the mesh with element's length.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        self.L = TYPEFLOAT(L)
        self.bar = np.nan
        self.dim = 1
        self.size = np.nan
        self.type = np.nan
        self.faces = np.nan
        self.faceslist = np.nan
        self.coord = np.nan
        self.point = np.nan
        self.sides = np.nan

    def translate(self, X):
        """

        Translate the initial mesh by a X vector.

        Parameters
        ----------
        X : ARRAY OF FLOAT or LIST OF FLOAT
            translation vector in (x, y) or (x, y, z) coordinates.

        Returns
        -------
        None.

        """
        if type(X) == list:
            X = np.array(X)
        else:
            pass

        self.point += X.reshape((1, len(X)))

    def rotate(self, theta):
        """

        Rotate the initial mesh around the z-axis by a theta angle.

        Parameters
        ----------
        theta : FLOAT
            angle of rotation for the mesh around the Z axis in 2D and 3D.

        Returns
        -------
        None.

        """
        matrix = np.identity(self.dim)
        matrix[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        self.point = self.point.dot(matrix)

    def __repr__(self):
        return "{}D regular mesh of {} points and {} sides.".format(self.dim,
                self.size, self.sides)


#%% Definition of the differents types of elementary meshes in 2D or 3D

class TriMesh2D(Mesh):
    """Define an aequilateral triangular mesh constructed with the length of one side."""

    def __init__(self, L):
        """

        Build the triangular mesh with the x, y coordinates of the 3 points.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 2
        self.size = 3       # number of points
        self.sides = 3      # number of sides
        self.coord = 6      # coordination number
        self.type = "Tri"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0]) * self.L
        self.point[1, :] = np.array([1, 0]) * self.L
        self.point[2, :] = np.array([0.5, (np.sqrt(3) / 2)]) * self.L

        self.faceslist[0, :] = np.array([0, 1])
        self.faceslist[1, :] = np.array([1, 2])
        self.faceslist[2, :] = np.array([2, 0])

        self.bar = np.sum(self.point, axis=0) / self.size
        self.dX = 0.5
        self.dY = np.sqrt(3) / 2


class TriDTMesh2D(Mesh):
    """Mimick an elementary mesh for Delaunay Triangulation-based lattice."""

    def __init__(self, L):
        """

        Build the Delaunay Triangulation mesh with a typical length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 2
        self.size = 0       # number of points
        self.sides = 0      # number of sides
        self.coord = 6      # coordination number
        self.type = "TriDT"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.bar = np.array([[0, 0]])
        self.dX = 0
        self.dY = 0


class VoronoiMesh2D(Mesh):
    """Mimick an elementary mesh for Voronoi-based lattice."""

    def __init__(self, L, startinglattice="Square"):
        """

        Build a mock mesh based on a real elementary mesh.

        Parameters
        ----------
        L : FLOAT
            beam's length.
        startinglattice : STRING, optional
            Starting mesh on wich you want to apply your voronoi tesselation.
            The default is "Square".

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 2
        self.size = 0       # number of points
        self.sides = 0      # number of sides
        self.coord = 4      # coordination number
        self.type = "Vor"
        self.startinglattice = startinglattice

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.bar = np.array([[0, 0]])
        self.dX = 0
        self.dY = 0


class SquareMesh2D(Mesh):
    """Build a square mesh constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the square mesh with the length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 2
        self.size = 4       # number of points
        self.sides = 4      # number of sides
        self.coord = 4      # coordination number
        self.type = "Square"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0]) * self.L
        self.point[1, :] = np.array([1, 0]) * self.L
        self.point[2, :] = np.array([1, 1]) * self.L
        self.point[3, :] = np.array([0, 1]) * self.L

        self.faceslist[0, :] = np.array([0, 1])
        self.faceslist[1, :] = np.array([1, 2])
        self.faceslist[2, :] = np.array([2, 3])
        self.faceslist[3, :] = np.array([3, 0])

        self.bar = np.sum(self.point, axis=0) / self.size
        self.dX = 1
        self.dY = 1


class HexaMesh2D(Mesh):
    """Object that defines a regular hexagonal mesh constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the hexagon mesh with the length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 2
        self.size = 6       # number of points
        self.sides = 6      # number of sides
        self.coord = 3      # coordination number
        self.type = "Hexa"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([-np.sqrt(3) / 2, -0.5]) * self.L
        self.point[1, :] = np.array([0, -1]) * self.L
        self.point[2, :] = np.array([np.sqrt(3) / 2, -0.5]) * self.L
        self.point[3, :] = np.array([np.sqrt(3) / 2, 0.5]) * self.L
        self.point[4, :] = np.array([0, 1]) * self.L
        self.point[5, :] = np.array([-np.sqrt(3) / 2, 0.5]) * self.L

        self.faceslist[0, :] = np.array([0, 1])
        self.faceslist[1, :] = np.array([1, 2])
        self.faceslist[2, :] = np.array([2, 3])
        self.faceslist[3, :] = np.array([3, 4])
        self.faceslist[4, :] = np.array([4, 5])
        self.faceslist[5, :] = np.array([5, 0])

        self.bar = np.sum(self.point, axis=0) / self.size
        self.dX = np.sqrt(3)
        self.dY = 1.5


class CubicMesh3D(Mesh):
    """Object that defines a cubic mesh constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the cube with the length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 8               # number of points
        self.sides = 12             # number of sides
        self.faces = 6              # number of faces
        self.coord = 6              # coordination number
        self.type = "Cubic"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.faces, 4), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0, 0]) * self.L
        self.point[1, :] = np.array([1, 0, 0]) * self.L
        self.point[2, :] = np.array([1, 1, 0]) * self.L
        self.point[3, :] = np.array([0, 1, 0]) * self.L
        self.point[4, :] = np.array([0, 0, 1]) * self.L
        self.point[5, :] = np.array([1, 0, 1]) * self.L
        self.point[6, :] = np.array([1, 1, 1]) * self.L
        self.point[7, :] = np.array([0, 1, 1]) * self.L

        self.faceslist[0, :] = np.array([0, 1, 2, 3])
        self.faceslist[1, :] = np.array([0, 1, 5, 4])
        self.faceslist[2, :] = np.array([1, 2, 6, 5])
        self.faceslist[3, :] = np.array([2, 3, 7, 6])
        self.faceslist[4, :] = np.array([3, 0, 4, 7])
        self.faceslist[5, :] = np.array([4, 5, 6, 7])

        self.bar = np.sum(self.point, axis=0) / self.size
        self.dX = 1
        self.dY = 1
        self.dZ = 1


class TriMesh3D(Mesh):
    """Object that defines a triangular prism given the length of one side."""

    def __init__(self, L):
        """

        Build the triangular prism with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 6               # number of points
        self.sides = 9              # number of sides
        self.faces = 5              # number of faces
        self.coord = 8              # coordination number
        self.type = "TriPrism"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.faces, 4), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0, 0]) * self.L
        self.point[1, :] = np.array([1, 0, 0]) * self.L
        self.point[2, :] = np.array([0.5, (np.sqrt(3) / 2), 0]) * self.L
        self.point[3, :] = np.array([0, 0, 1]) * self.L
        self.point[4, :] = np.array([1, 0, 1]) * self.L
        self.point[5, :] = np.array([0.5, (np.sqrt(3) / 2), 1]) * self.L

        self.faceslist[0, :] = np.array([0, 1, 2, np.nan])
        self.faceslist[1, :] = np.array([0, 1, 4, 3])
        self.faceslist[2, :] = np.array([1, 2, 5, 4])
        self.faceslist[3, :] = np.array([2, 0, 3, 5])
        self.faceslist[4, :] = np.array([3, 4, 5, np.nan])

        self.bar = np.sum(self.point, axis=0) / self.size


class TriDTMesh3D(Mesh):
    """Object use to mimick an elementary mesh for Delaunay Triangulation-based lattice in 3D."""

    def __init__(self, L):
        """

        Build the Delaunay Triangulation mesh with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 0       # number of points
        self.sides = 0      # number of sides
        self.faces = 0      # number of faces
        self.coord = 14      # coordination number
        self.type = "TriDT3D"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.bar = np.array([[0, 0, 0]])
        self.dX = 0
        self.dY = 0
        self.dZ = 0




class VoronoiMesh3D(Mesh):
    """Object use to mimick an elementary mesh for Voronoi-based lattice in 3D."""

    def __init__(self, L, startinglattice="Cubic"):
        """

        Build a mock mesh with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.
        startinglattice : STRING, optional
            Starting mesh on wich you want to apply your voronoi tesselation.
            The default is "Cubic".

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 0       # number of points
        self.sides = 0      # number of sides
        self.faces = 0      # number of faces
        self.coord = 0      # coordination number
        self.type = "Vor3D"
        self.startinglattice = startinglattice

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.sides, 2), dtype=TYPEFLOAT)

        self.bar = np.array([[0, 0, 0]])
        self.dX = 0
        self.dY = 0
        self.dZ = 0




class HexaMesh3D(Mesh):
    """Object that defines a hexagonal prism constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the hexagonal prism with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 12              # number of points
        self.sides = 18             # number of sides
        self.faces = 8              # number of faces
        self.coord = 5              # coordination number
        self.type = "HexaPrism"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.faces, 6), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0.5, 0]) * self.L
        self.point[1, :] = np.array([(np.sqrt(3) / 2), 0, 0]) * self.L
        self.point[2, :] = np.array([(2 * np.sqrt(3) / 2), 0.5, 0]) * self.L
        self.point[3, :] = np.array([(2 * np.sqrt(3) / 2), 1.5, 0]) * self.L
        self.point[4, :] = np.array([(np.sqrt(3) / 2), 2, 0]) * self.L
        self.point[5, :] = np.array([0, 1.5, 0]) * self.L
        self.point[6, :] = np.array([0, 0.5, 1]) * self.L
        self.point[7, :] = np.array([(np.sqrt(3) / 2), 0, 1]) * self.L
        self.point[8, :] = np.array([(2 * np.sqrt(3) / 2), 0.5, 1]) * self.L
        self.point[9, :] = np.array([(2 * np.sqrt(3) / 2), 1.5, 1]) * self.L
        self.point[10, :] = np.array([(np.sqrt(3) / 2), 2, 1]) * self.L
        self.point[11, :] = np.array([0, 1.5, 1]) * self.L

        self.faceslist[0, :] = np.array([0, 1, 2, 3, 4, 5])
        self.faceslist[1, :] = np.array([0, 1, 7, 6, np.nan, np.nan])
        self.faceslist[2, :] = np.array([1, 2, 8, 7, np.nan, np.nan])
        self.faceslist[3, :] = np.array([2, 3, 9, 8, np.nan, np.nan])
        self.faceslist[4, :] = np.array([3, 4, 10, 9, np.nan, np.nan])
        self.faceslist[5, :] = np.array([4, 5, 11, 10, np.nan, np.nan])
        self.faceslist[6, :] = np.array([5, 0, 6, 11, np.nan, np.nan])
        self.faceslist[7, :] = np.array([6, 7, 8, 9, 10, 11])

        self.bar = np.sum(self.point, axis=0) / self.size


class RhombicDodecahedron3D(Mesh):
    """Object that defines a rhombic dodecahedron constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the rhombic dodecahedron with the the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 14               # number of points
        self.sides = 24              # number of sides
        self.faces = 12              # number of faces
        self.coord = 8               # coordination number
        self.type = "Dodec"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.faces, 4), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([0, 0, 2]) / np.sqrt(3) * self.L
        self.point[1, :] = np.array([-np.sqrt(2), 0, 1]) / np.sqrt(3) * self.L
        self.point[2, :] = np.array([-np.sqrt(2), -np.sqrt(2), 0]) / np.sqrt(3) * self.L
        self.point[3, :] = np.array([0, -np.sqrt(2), 1]) / np.sqrt(3) * self.L
        self.point[4, :] = np.array([0, np.sqrt(2), 1]) / np.sqrt(3) * self.L
        self.point[5, :] = np.array([-np.sqrt(2), np.sqrt(2), 0]) / np.sqrt(3) * self.L
        self.point[6, :] = np.array([-np.sqrt(2), 0, -1]) / np.sqrt(3) * self.L
        self.point[7, :] = np.array([np.sqrt(2), 0, 1]) / np.sqrt(3) * self.L
        self.point[8, :] = np.array([np.sqrt(2), -np.sqrt(2), 0]) / np.sqrt(3) * self.L
        self.point[9, :] = np.array([np.sqrt(2), np.sqrt(2), 0]) / np.sqrt(3) * self.L
        self.point[10, :] = np.array([np.sqrt(2), 0, -1]) / np.sqrt(3) * self.L
        self.point[11, :] = np.array([0, np.sqrt(2), -1]) / np.sqrt(3) * self.L
        self.point[12, :] = np.array([0, 0, -2]) / np.sqrt(3) * self.L
        self.point[13, :] = np.array([0, -np.sqrt(2), -1]) / np.sqrt(3) * self.L

        self.faceslist[0, :] = np.array([0, 1, 2, 3])
        self.faceslist[1, :] = np.array([0, 1, 5, 4])
        self.faceslist[2, :] = np.array([1, 2, 6, 5])
        self.faceslist[3, :] = np.array([0, 3, 8, 7])
        self.faceslist[4, :] = np.array([0, 4, 9, 7])
        self.faceslist[5, :] = np.array([7, 8, 10, 9])
        self.faceslist[6, :] = np.array([2, 3, 8, 13])
        self.faceslist[7, :] = np.array([4, 5, 11, 9])
        self.faceslist[8, :] = np.array([2, 6, 12, 13])
        self.faceslist[9, :] = np.array([6, 5, 11, 12])
        self.faceslist[10, :] = np.array([13, 12, 10, 8])
        self.faceslist[11, :] = np.array([11, 12, 10, 9])

        self.bar = np.sum(self.point, axis=0) / self.size


class TruncatedOctahedron3D(Mesh):
    """Object that defines a truncated octahedron constructedgiven the length of one side."""

    def __init__(self, L):
        """

        Build the truncated octahedron with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 24               # number of points
        self.sides = 36              # number of sides
        self.faces = 14              # number of faces
        self.coord = 4               # coordination number
        self.type = "Octa"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = np.zeros((self.faces, 6), dtype=TYPEFLOAT)

        self.point[0, :] = np.array([np.sqrt(2), np.sqrt(2) / 2, 0]) * self.L
        self.point[1, :] = np.array([np.sqrt(2), 0, np.sqrt(2) / 2]) * self.L
        self.point[2, :] = np.array([np.sqrt(2), -np.sqrt(2) / 2, 0]) * self.L
        self.point[3, :] = np.array([np.sqrt(2), 0, -np.sqrt(2) / 2]) * self.L
        self.point[4, :] = np.array([0, np.sqrt(2), np.sqrt(2) / 2]) * self.L
        self.point[5, :] = np.array([np.sqrt(2) / 2, np.sqrt(2), 0]) * self.L
        self.point[6, :] = np.array([0, np.sqrt(2), -np.sqrt(2) / 2]) * self.L
        self.point[7, :] = np.array([-np.sqrt(2) / 2, np.sqrt(2), 0]) * self.L
        self.point[8, :] = np.array([np.sqrt(2) / 2, 0, np.sqrt(2)]) * self.L
        self.point[9, :] = np.array([0, np.sqrt(2) / 2, np.sqrt(2)]) * self.L
        self.point[10, :] = np.array([-np.sqrt(2) / 2, 0, np.sqrt(2)]) * self.L
        self.point[11, :] = np.array([0, -np.sqrt(2) / 2, np.sqrt(2)]) * self.L
        self.point[12, :] = np.array([-np.sqrt(2), 0, -np.sqrt(2) / 2]) * self.L
        self.point[13, :] = np.array([-np.sqrt(2), -np.sqrt(2) / 2, 0]) * self.L
        self.point[14, :] = np.array([-np.sqrt(2), 0, np.sqrt(2) / 2]) * self.L
        self.point[15, :] = np.array([-np.sqrt(2), np.sqrt(2) / 2, 0]) * self.L
        self.point[16, :] = np.array([-np.sqrt(2) / 2, -np.sqrt(2), 0]) * self.L
        self.point[17, :] = np.array([0, -np.sqrt(2), -np.sqrt(2) / 2]) * self.L
        self.point[18, :] = np.array([np.sqrt(2) / 2, -np.sqrt(2), 0]) * self.L
        self.point[19, :] = np.array([0, -np.sqrt(2), np.sqrt(2) / 2]) * self.L
        self.point[20, :] = np.array([0, -np.sqrt(2) / 2, -np.sqrt(2)]) * self.L
        self.point[21, :] = np.array([-np.sqrt(2) / 2, 0, -np.sqrt(2)]) * self.L
        self.point[22, :] = np.array([0, np.sqrt(2) / 2, -np.sqrt(2)]) * self.L
        self.point[23, :] = np.array([np.sqrt(2) / 2, 0, -np.sqrt(2)]) * self.L

        self.faceslist[0, :] = np.array([0, 1, 2, 3, np.nan, np.nan])
        self.faceslist[1, :] = np.array([4, 5, 6, 7, np.nan, np.nan])
        self.faceslist[2, :] = np.array([8, 9, 10, 11, np.nan, np.nan])
        self.faceslist[3, :] = np.array([12, 13, 14, 15, np.nan, np.nan])
        self.faceslist[4, :] = np.array([16, 17, 18, 19, np.nan, np.nan])
        self.faceslist[5, :] = np.array([20, 21, 22, 23, np.nan, np.nan])
        self.faceslist[6, :] = np.array([0, 1, 8, 9, 4, 5])
        self.faceslist[7, :] = np.array([1, 2, 18, 19, 11, 8])
        self.faceslist[8, :] = np.array([2, 3, 23, 20, 17, 18])
        self.faceslist[9, :] = np.array([3, 0, 5, 6, 22, 23])
        self.faceslist[10, :] = np.array([12, 13, 16, 17, 20, 21])
        self.faceslist[11 :] = np.array([13, 14, 10, 11, 19, 16])
        self.faceslist[12, :] = np.array([14, 15, 7, 4, 9, 10])
        self.faceslist[13, :] = np.array([15, 12, 21, 22, 6, 7])

        self.bar = np.sum(self.point, axis=0) / self.size


class OctetTruss3D(Mesh):
    """Object that defines an octet-truss constructed given the length of one side."""

    def __init__(self, L):
        """

        Build the octet-truss with the side length L.

        Parameters
        ----------
        L : FLOAT
            beam's length.

        Returns
        -------
        None.

        """
        Mesh.__init__(self, L)
        self.dim = 3
        self.size = 14              # number of points
        self.sides = 36             # number of sides
        self.faces = [6, 16]        # number of faces
        self.coord = 12             # coordination number
        self.type = "Octet"

        self.point = np.zeros((self.size, self.dim), dtype=TYPEFLOAT)
        self.faceslist = [np.zeros((6, 3), dtype=TYPEFLOAT), np.zeros((16, 4), dtype=TYPEFLOAT)]

        self.point[0, :] = np.array([0, 0, 0]) * self.L * np.sqrt(2)
        self.point[1, :] = np.array([1, 0, 0]) * self.L * np.sqrt(2)
        self.point[2, :] = np.array([1, 1, 0]) * self.L * np.sqrt(2)
        self.point[3, :] = np.array([0, 1, 0]) * self.L * np.sqrt(2)
        self.point[4, :] = np.array([0, 0, 1]) * self.L * np.sqrt(2)
        self.point[5, :] = np.array([1, 0, 1]) * self.L * np.sqrt(2)
        self.point[6, :] = np.array([1, 1, 1]) * self.L * np.sqrt(2)
        self.point[7, :] = np.array([0, 1, 1]) * self.L * np.sqrt(2)
        self.point[8, :] = np.array([0.5, 0.5, 0]) * self.L * np.sqrt(2)
        self.point[9, :] = np.array([0.5, 0.5, 1]) * self.L * np.sqrt(2)
        self.point[10, :] = np.array([0.5, 0, 0.5]) * self.L * np.sqrt(2)
        self.point[11, :] = np.array([0.5, 1, 0.5]) * self.L * np.sqrt(2)
        self.point[12, :] = np.array([0, 0.5, 0.5]) * self.L * np.sqrt(2)
        self.point[13, :] = np.array([1, 0.5, 0.5]) * self.L * np.sqrt(2)

        self.faceslist[0][0, :] = np.array([0, 1, 2])
        self.faceslist[0][1, :] = np.array([4, 5, 6])
        self.faceslist[0][2, :] = np.array([0, 1, 5])
        self.faceslist[0][3, :] = np.array([2, 3, 7])
        self.faceslist[0][4, :] = np.array([1 ,2, 6])
        self.faceslist[0][5, :] = np.array([0, 3, 7])

        self.faceslist[1][0, :] = np.array([8, 10, 13, np.nan])
        self.faceslist[1][1, :] = np.array([8, 13, 11, np.nan])
        self.faceslist[1][2, :] = np.array([8, 11, 12, np.nan])
        self.faceslist[1][3, :] = np.array([8, 12, 10, np.nan])
        self.faceslist[1][4, :] = np.array([9, 10, 13, np.nan])
        self.faceslist[1][5, :] = np.array([9, 13, 11, np.nan])
        self.faceslist[1][6, :] = np.array([9, 11, 12, np.nan])
        self.faceslist[1][7, :] = np.array([9, 12, 10, np.nan])
        self.faceslist[1][8, :] = np.array([9, 5, 13, 6])
        self.faceslist[1][9, :] = np.array([1, 13, 2, 8])
        self.faceslist[1][10, :] = np.array([9, 4, 12, 7])
        self.faceslist[1][11, :] = np.array([0, 12, 3, 8])
        self.faceslist[1][12, :] = np.array([11, 6, 9, 7])
        self.faceslist[1][13, :] = np.array([11, 2, 8, 3])
        self.faceslist[1][14, :] = np.array([9, 4, 10, 5])
        self.faceslist[1][15, :] = np.array([8, 1, 10, 0])

        self.bar = np.sum(self.point, axis=0) / self.size
        self.dX = 1
        self.dY = 1
        self.dZ = 1

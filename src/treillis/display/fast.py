"""
Fast display
============

Interactive 2D and 3D graphs of the lattices. Quality may not be adapted to
publication.

This submodule is built using Datoviz, developed by Cyrille Rossant, and is
bound to evolve as the library grows. Once VisPy 2.0 is completed,
`fast display` may see a full rehaul, and hopefully easier figure completion
by users.
"""

#%% Importation of useful packages
from copy import deepcopy

import numpy as np
import matplotlib.colors as mc
import matplotlib as mpl
import datoviz as dvz

import treillis

__all__ = ['showlat', 'shownodes', 'previz',
           'colorlat', 'displacement', 'scatter_lat', 'slices']

from . import plot_utilitaries as pu

xplot = int(600)
yplot = int(600)

RED = '#E50019'
DARK_BLUE = '#3E4A83'
DARK_GREY = '#1E1E1E'
LIGHT_BLUE = '#7E9CBB'
YELLOW = '#FFCD31'
MACARON = '#DA837B'
ARCHIPEL = '#00939D'
OPERA = '#BD987A'
WISTERIA = '#A72587'

rainbow = [YELLOW, OPERA, MACARON, RED, WISTERIA, DARK_BLUE, LIGHT_BLUE, 
           ARCHIPEL]
rmap = pu.make_cmap(rainbow)

#%% useful fonctions for plotting
def signif(app, data):
    """Create xticks from data in scientific notation with a precision of 2."""
    # Find common order of magnitude to have most tick in format 1.*f
    data=data.ravel()
    data = np.linspace(min(data), max(data), num=7)

    exp = np.floor(np.log10(np.delete(abs(data),
                                      np.arange(len(data))[data==0])))
    exp = np.unique(exp)
    if len(exp)==1:
        oom = int(exp[0])
    elif len(exp)==2:
        oom = int(max(exp))
    elif len(exp)>2:
        oom = int(max(exp))-1
    elif len(exp)==0:
        oom=0

    tick = ['%.2f'%(d/10**oom) for d in data]
    tick.append('x 10')
    tick.append(str(oom))
    strscale = np.array([1]*8+[.6], dtype=np.float32)

    ticks = app.glyph(font_size=20)

    Y = np.array([-.4]*7).reshape((7,1))
    X=np.linspace(-.75,.75,7).reshape((7,1))
    Z = np.array([0]*7).reshape((7,1))
    xyz = np.concatenate((X,Y,Z), axis=1, dtype=np.float32)

    xyz = np.concatenate((xyz, np.array([[.73, .4, 0], [.83,.5, 0]])),
                         axis=0, dtype=np.float32)

    ticks.set_strings(tick, string_pos=xyz, color=(0,0,0,255),
                      scales=strscale)

    return ticks


def draw_edges(app, dim, size, nodes, elems, colors, depth=2):
    """Generate the segment graph of the lattice with the given edge colors."""
    X_in = nodes[elems[:, 0].astype(int), 0]
    X_in = X_in.reshape(np.shape(elems)[0], 1, 1)
    Y_in = nodes[elems[:, 0].astype(int), 1]
    Y_in = Y_in.reshape(np.shape(elems)[0], 1, 1)
    X_out = nodes[elems[:, 1].astype(int), 0]
    X_out = X_out.reshape(np.shape(elems)[0], 1, 1)
    Y_out = nodes[elems[:, 1].astype(int), 1]
    Y_out = Y_out.reshape(np.shape(elems)[0], 1, 1)
    if dim == 3:
        Z_in = nodes[elems[:, 0].astype(int), 2]
        Z_in = Z_in.reshape(np.shape(elems)[0], 1, 1)
        Z_out = nodes[elems[:, 1].astype(int), 2]
        Z_out = Z_out.reshape(np.shape(elems)[0], 1, 1)

    X_line = np.concatenate((X_in, X_out), axis=1)
    Y_line = np.concatenate((Y_in, Y_out), axis=1)
    Z_line = np.zeros(np.shape(X_line))
    lines = np.concatenate((X_line, Y_line, Z_line), axis=2)
    if dim == 3:
        Z_line = np.concatenate((Z_in, Z_out), axis=1)
        lines = np.concatenate((X_line, Y_line, Z_line), axis=2)

    if isinstance(colors, tuple): colors=[colors]*len(X_line)

    seg = app.segment(initial=lines[:,0,:], terminal=lines[:,1,:],linewidth=3.5,
                      color=colors,
                      cap=('triangle_out','triangle_out'),
                      depth_test=(depth==2)*(dim==3)+depth*(depth!=2))
    return seg


def plotBox(dim, size, app):
    """Plot a *box* domain."""
    if len(size) == 1 and dim == 2:
        boxpath = app.marker(position=np.array([0,0,0]).reshape(1,3),
                             edgecolor=(0,0,0,100),
                             size=size, linewidth=3)
        boxpath.set_mode('code')
        boxpath.set_aspect('stroke')
        boxpath.set_shape('disc')


    elif len(size) == 1 and dim == 3:
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
        x = size[0] * np.cos(u)*np.sin(v)
        y = size[0] * np.sin(u)*np.sin(v)
        z = size[0] * np.cos(v)
        x1 = x.ravel()
        x1 = x1.reshape((len(x1),1))
        y1 = y.ravel()
        y1 = y1.reshape((len(y1),1))
        z1 = z.ravel()
        z1 = z1.reshape((len(z1),1))

        xyz1 = np.concatenate((x1,y1,z1), axis=1)

        x2 = x.T.ravel()
        x2 = x2.reshape((len(x2),1))
        y2 = y.T.ravel()
        y2 = y2.reshape((len(y2),1))
        z2 = z.T.ravel()
        z2 = z2.reshape((len(z2),1))

        xyz2 = np.concatenate((x2,y2,z2), axis=1)

        xyz = np.concatenate((xyz1,xyz2), axis=0, dtype=np.float32)
        boxpath = app.path()
        boxpath.set_position(position = xyz, groups=20)
        boxpath.set_data(color=[(0,0,0,100)]*np.shape(xyz)[0], linewidth=3,
                         depth_test=True)

    elif len(size) == 2:
        boxpath = app.path()
        boxpath.set_position(position = np.array([[-size[0]/2, -size[1]/2, 0],
                                       [size[0]/2, -size[1]/2, 0],
                                       [size[0]/2, size[1]/2, 0],
                                       [-size[0]/2, size[1]/2, 0],
                                       [-size[0]/2, -size[1]/2, 0]]), groups=1)
        boxpath.set_data(color=np.array([[0,0,0,100]]*5), linewidth=3,
                         depth_test=True)
    elif len(size) == 3:
        x = np.array([[-size[0]/2],  [size[0]/2], [size[0]/2], [-size[0]/2],
                      [-size[0]/2],
                      [-size[0]/2], [size[0]/2], [size[0]/2], [-size[0]/2],
                      [-size[0]/2],
                      [-size[0]/2], [-size[0]/2], [-size[0]/2], [-size[0]/2],
                      [-size[0]/2],
                      [size[0]/2], [size[0]/2], [size[0]/2], [size[0]/2],
                      [size[0]/2]])

        y = np.array([[size[1]/2],  [size[1]/2], [-size[1]/2], [-size[1]/2],
                      [size[1]/2],
                      [size[1]/2],  [size[1]/2], [-size[1]/2], [-size[1]/2],
                      [size[1]/2],
                      [-size[1]/2],  [size[1]/2], [size[1]/2], [-size[1]/2],
                      [-size[1]/2],
                      [-size[1]/2],  [size[1]/2], [size[1]/2], [-size[1]/2],
                      [-size[1]/2]])

        z = np.array([[-size[2]/2],  [-size[2]/2], [-size[2]/2], [-size[2]/2],
                      [-size[2]/2],
                      [size[2]/2],  [size[2]/2], [size[2]/2], [size[2]/2],
                      [size[2]/2],
                      [-size[2]/2],  [-size[2]/2], [size[2]/2], [size[2]/2],
                      [-size[2]/2],
                      [-size[2]/2],  [-size[2]/2], [size[2]/2], [size[2]/2],
                      [-size[2]/2]])
        xyz = np.concatenate((x,y,z), axis=1)
        boxpath = app.path()
        boxpath.set_position(position = xyz, groups=4)
        boxpath.set_data(color=[(0,0,0,100)]*np.shape(xyz)[0], linewidth=3,
                         depth_test=True)

    elif len(size) == 5 and dim == 2:
        boxpath = app.path(position =[[-size[0]/2, -size[1]/2, 0],
                                      [size[0]/2, -size[1]/2, 0],
                        [size[0]/2, size[1]/2, 0],
                        [-size[0]/2, size[1]/2, 0],
                       [-size[0]/2, size[3]/2, 0],
                       [-size[0]/2 + size[4], size[3]/2, 0],
                       [-size[0]/2 + size[4], -size[3]/2, 0],
                       [-size[0]/2, -size[3]/2, 0]],
                           color = [(0,0,0,100)]*9, linewidth=3,
                                            depth_test=True, groups=1)

    elif len(size) == 5 and dim == 3:
        pos = []
        col = []
        pos.append(np.array([[-size[0]/2, -size[1]/2, -size[2]/2],
                                      [size[0]/2, -size[1]/2, -size[2]/2],
                        [size[0]/2, size[1]/2, -size[2]/2],
                        [-size[0]/2, size[1]/2,-size[2]/2],
                       [-size[0]/2, size[3]/2, -size[2]/2],
                       [-size[0]/2 + size[4], size[3]/2,-size[2]/2],
                       [-size[0]/2 + size[4], -size[3]/2, -size[2]/2],
                       [-size[0]/2, -size[3]/2, -size[2]/2],
                       [-size[0]/2, -size[1]/2, -size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]

        pos.append(np.array([[-size[0]/2, -size[1]/2, size[2]/2],
                                      [size[0]/2, -size[1]/2, size[2]/2],
                        [size[0]/2, size[1]/2, size[2]/2],
                        [-size[0]/2, size[1]/2,size[2]/2],
                       [-size[0]/2, size[3]/2, size[2]/2],
                       [-size[0]/2 + size[4], size[3]/2,size[2]/2],
                       [-size[0]/2 + size[4], -size[3]/2, size[2]/2],
                       [-size[0]/2, -size[3]/2, size[2]/2],
                       [-size[0]/2, -size[1]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]

        pos.append(np.array([[-size[0]/2, -size[1]/2, -size[2]/2],
                             [-size[0]/2, -size[1]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[size[0]/2, -size[1]/2, -size[2]/2],
                             [size[0]/2, -size[1]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[-size[0]/2, size[1]/2, -size[2]/2],
                             [-size[0]/2, size[1]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[size[0]/2, size[1]/2, -size[2]/2],
                             [size[0]/2, size[1]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]

        pos.append(np.array([[-size[0]/2, size[3]/2, -size[2]/2],
                             [-size[0]/2, size[3]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[-size[0]/2, -size[3]/2, -size[2]/2],
                             [-size[0]/2, -size[3]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[size[4]-size[0]/2, size[3]/2, -size[2]/2],
                             [size[4]-size[0]/2, size[3]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]
        pos.append(np.array([[size[4]-size[0]/2, -size[3]/2, -size[2]/2],
                             [size[4]-size[0]/2, -size[3]/2, size[2]/2]]))
        col+=[(0,0,0,100)]*np.shape(pos[-1])[0]

        boxpath = app.path(pos, groups=1)
        boxpath.set_data(color=col,
                         linewidth=3, depth_test=True)
    else:
        raise ValueError('Other kind of domains are not implemented. Feel free to add yours ;)')

    return boxpath


def plotAxis(dim, size, app, panel):
    """Show the x, y (and z if 3D) axes."""

    if dim == 2:
        if len(size) == 1:
            delta = 0.1 * size[0].item() * 2
        else:
            delta = max(0.1 * size[0].item(), 0.1 * size[1].item())
        lim_z = [0, 0]
        Z = lim_z[0]

    elif dim ==3:
        if len(size) == 1:
            delta = 0.1 * size[0].item() * 2
        else:
            delta = max(0.1 * size[0].item(), 0.1 * size[1].item(), 0.1 * size[2].item())

    if len(size)==1: size = [size[0].item(),size[0].item(),size[0].item()]
    lim_x, lim_y = [-size[0]/2, size[0]/2], [-size[1]/2, size[1]/2]

    X = lim_x[0]
    Y = lim_y[0]
    if dim==3:
        lim_z = [-size[2]/2, size[2]/2]
        Z = lim_z[0]

    seg0 = np.array([[X - 2 * delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X + delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X + delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y + delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y + delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y - 2 * delta/3, Z + delta/3*(Z!=0)],
            [X - 2 * delta/3, Y - 2 * delta/3, Z + delta/3*(Z!=0)]])

    segf = np.array([[X + delta/3, Y - 2 * delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y + delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3, Y - 2 * delta/3, Z + delta/3*(Z!=0)],
            [X , Y - delta, Z - 2 * delta/3*(Z!=0)],
            [X , Y - delta/3, Z - 2 * delta/3*(Z!=0)],
            [X - delta, Y, Z - 2 * delta/3*(Z!=0)],
            [X - delta/3, Y, Z - 2 * delta/3*(Z!=0)],
            [X - 2 * delta/3 - (Z!=0) * delta/3, Y - 2 * delta/3, Z],
            [X - 2 * delta/3 + (Z!=0) * delta/3, Y - 2 * delta/3, Z]])

    axpath = app.segment(initial = seg0, terminal = segf,
                         color = [(4, 163, 49, 255)]*len(seg0), linewidth=5,
                         cap = ('triangle_out', 'triangle_out'),
                                          depth_test=True)
    panel.add(axpath)

    xyz = app.glyph(font_size=13)
    if dim==3:
        xyz.set_strings(
            strings=['x', 'y', 'z'],
            string_pos=np.array([[X + 1.5*delta/3, Y - 2 * delta/3, Z-2*delta/3],
                        [X - 2 * delta/3, Y + 1.5 * delta/3, Z-2*delta/3],
                        [X - 2 * delta/3, Y-2*delta/3, Z + 1.5 * delta/3]],
                                dtype=np.float32),
            scales=np.array([2,2,2], dtype=np.float32))
        xyz.set_color([(0, 0, 0, 255)]*3)
    else:
        xyz.set_strings(
            strings=['x', 'y'],
            string_pos=np.array([[X - 1.5 * delta, Y - 0.2 * delta, Z ],
                        [X - 0.2 * delta, Y - 1.5 * delta, Z ]],
                                dtype=np.float32),
            scales=np.array([2,2], dtype=np.float32),)
        xyz.set_color([(0, 0, 0, 255)]*2)
    panel.add(xyz)


def initfig(nx=1, ny=1):
    """Initialize the applet for fast_display."""
    app = dvz.App(background='white')
    fig = app.figure(int(nx*xplot),int(ny*yplot))
    return fig, app


def colorbar(app, figure, data, label, x=0,y=1, nx=1, ny=1, cmap='jet'):
    """Create a colorbar panel to describe the figure."""
    cbar = figure.panel((x,ny*yplot),(nx*xplot,yplot/3/y))
    visu = app.path()
    visu.set_position(np.array([[i,0,0] for i in np.linspace(-.75,.75,100)],
                               dtype=np.float32), groups=1)
    visu.set_data(color=dvz.cmap(cmap, np.arange(100), vmin=0, vmax=100),
                  linewidth=30, cap='square')
    cbar.add(visu)
    cbar.add(signif(app, data))
    L = len(label)
    if L>0:
        legend = app.glyph(font_size=30)
        legend.set_strings([label],
                          string_pos=np.array([[0,.55,0] for i in\
                                               np.linspace(.42,-.42,L)],
                                              dtype=np.float32).reshape((L,3)),
                          scales=np.array([1]*L, dtype=np.float32),
                          color=(0,0,0,255))
        legend.set_angle(np.array([-np.pi/2]*L, dtype=np.float32))
        cbar.add(legend)
        
        
def color_depth(color):
    clist = [pu.adjust_lightness(color, i) 
            for i in np.linspace(1.4,.7,20)*.5]
    cmap = pu.make_cmap(clist)
    return cmap


#%% Plotting functions
#%%% Basic visualization
# simple lattice
def showlat(lattice, box = True, axis = True,
            boundary = False, color = RED, in_app=False):
    """
    Show the lattice.

    Possibility to add the xyz axes as well as the box domain visualization.
    The Datoviz app and panel can be yielded to add other visualizations
    outside of the function.

    Parameters
    ----------
    lattice : Lattice-like object
        Lattice to be shown
    box : BOOL, optional
        if you want to display the domain. The default is True.
    axis : BOOL, optional
        if you want to add an axis system. The default is True.
    boundary : BOOL, optional
        if you want to display in green the nodes at the boundary. The default is False.
    color : optional
        color of the segments, by default blood red.
    in_app : BOOL, optional
        whether to use the base lattice figure and add other visuals or to directly plot it.

    Returns
    -------
    if in_app fig, panel, app : DATOVIZ FIGURE, PANEL AND APP
        allows to add additional visuals to the panel before showing.

    """
    dim = lattice.dim
    size = lattice.sizedom[:lattice.dim]
    norm = np.sqrt(sum(size**2))/dim
    if len(size)==1 : norm = np.sqrt(dim*size**2)/dim
    if len(size)>2 and size[2]==0: dim = 2
    nodes = deepcopy(lattice.nodes)/norm
    elems = lattice.elems
    if len(size)>1: r=size[1]/size[0]
    else: r=1
    
    # Create the figure
    fig1, app = initfig(ny=r)
    panel = fig1.panel()
    if dim==2: panel.panzoom()
    else: panel.arcball()
    
    D = np.sqrt(np.sum(
        (lattice.nodes[lattice.elems[:,0].astype(int)]
         + lattice.nodes[lattice.elems[:,1].astype(int)])**2 / size**2,
        axis=1))
    Norm = mc.Normalize(vmin=0, vmax=1)
    cmap = color_depth(color)
    colors = [[int(mc.to_rgba(c)[0]*255), int(mc.to_rgba(c)[1]*255), 
               int(mc.to_rgba(c)[2]*255), int(mc.to_rgba(c)[3]*255)]
              for c in cmap(Norm(D))]

    seg = draw_edges(app,dim,size,nodes,elems,colors)
    panel.add(seg)

    if boundary:
        X = nodes[lattice.fixed_nodes, 0].reshape((len(lattice.fixed_nodes),1))
        Y = nodes[lattice.fixed_nodes, 1].reshape((len(lattice.fixed_nodes),1))
        if dim==2:
            Z = np.zeros(np.shape(X))
        else:
            Z = nodes[lattice.fixed_nodes, 2].\
                reshape((len(lattice.fixed_nodes),1))

        XYZ = np.concatenate((X,Y,Z), axis=1)
        lim = app.sphere(position=XYZ,
                        color=[[0, 147, 157, 255]]*len(X),
                        size=lattice.L/lattice.aspect_ratio/15,
                                         depth_test=True)
        panel.add(lim)

    if box:
        bplot = plotBox(dim, size/norm, app)
        panel.add(bplot)

    if axis:
        plotAxis(dim, size/norm, app, panel)

    if in_app:
        return fig1, panel, app
    else:
        app.run()
        app.destroy()


def shownodes(lattice, box = True, axis = True,
            boundary = False, color: tuple|list = (138,3,3,255), in_app=False):
    """
    Show the lattice nodes without the elements.

    Parameters
    ----------
    lattice : Lattice-like object
        Lattice whose nodes you want to see.
    box : BOOL, optional
        if you want to display the domain. The default is True.
    axis : BOOL, optional
        if you want to add an axis system. The default is True.
    boundary : BOOL, optional
        if you want to display in green the nodes at the boundary. The default is False.
    color : RGBA-TUPLE, optional
        color of the points, by default blood red.
    in_app : BOOL, optional
        whether to use the base lattice figure and add other visuals or to directly plot it.

    Returns
    -------
    panel : TYPE
        DESCRIPTION.
    app : TYPE
        DESCRIPTION.

    """
    dim = lattice.dim
    size = lattice.sizedom[:lattice.dim]
    norm = np.sqrt(sum(size**2))/dim
    if len(size)==1 : norm = np.sqrt(dim*size**2)/dim
    if len(size)>2 and size[2]==0: dim = 2
    nodes = deepcopy(lattice.nodes)/norm
    if isinstance(color, tuple): color = [color]*len(nodes)
    if len(size)>1: r=size[1]/size[0]
    else: r=1
    # Create the figure
    fig1, app = initfig(ny=r)
    panel = fig1.panel()
    if dim==2: panel.panzoom()
    else: panel.arcball()

    dots = app.point(position=nodes, color=color, size=5, depth_test=True)
    panel.add(dots)

    if boundary:
        X = nodes[lattice.fixed_nodes, 0].reshape((len(lattice.fixed_nodes),1))
        Y = nodes[lattice.fixed_nodes, 1].reshape((len(lattice.fixed_nodes),1))
        if dim==2:
            Z = np.zeros(np.shape(X))
        else:
            Z = nodes[lattice.fixed_nodes, 2].\
                reshape((len(lattice.fixed_nodes),1))

        XYZ = np.concatenate((X,Y,Z), axis=1)
        lim = app.sphere(position=XYZ,
                        color=[[0, 147, 157, 255]]*len(X),
                        size=lattice.L/lattice.aspect_ratio/20,
                                         depth_test=True)
        panel.add(lim)

    if box:
        bplot = plotBox(dim, size/norm, app)
        panel.add(bplot)

    if axis:
        plotAxis(dim, size/norm, app, panel)

    if in_app:
        return fig1, panel, app
    else:
        app.run()
        app.destroy()





def previz(lattice: treillis.Lattice, color = (126,156,187,255), in_app=False):
    """Show a previzualisation of the lattice for 3D printing. It is recommended to only show a slice of the lattice."""
    nodes = deepcopy(lattice.nodes)
    
    length = lattice.L

    norm = norm = np.sqrt(np.sum(
        (np.max(nodes, axis=0)-np.min(nodes, axis=0))**2)
        )/lattice.dim
    if len(lattice.sizedom)>1: r=lattice.sizedom[1]/lattice.sizedom[0]
    else: r=1
    nodes, length = nodes/norm, length/norm
    fig, app = initfig(ny=r)
    pan = fig.panel()

    if lattice.dim==2: pan.panzoom()
    else: pan.arcball()
    sc = dvz.ShapeCollection()

    if lattice.dim==2:
        nodes = np.concatenate((nodes, np.zeros((nodes.shape[0],1))), axis=1)

    for i in range(len(lattice.elems)):
        elm = deepcopy(lattice.elems[i])
        seg_init = nodes[elm[0].astype(int),:]
        seg_fin = nodes[elm[1].astype(int),:]

        vec = seg_fin - seg_init
        xyz = (seg_init+seg_fin) / 2

        l = np.sqrt((seg_init[0]-seg_fin[0])**2 + (seg_init[1]-seg_fin[1])**2 \
                    + (seg_init[2]-seg_fin[2])**2)

        vec_init = np.array([0 , 1, 0])

        axis = np.cross(vec_init, vec)
        nvec = np.sqrt(np.sum(axis**2))
        if nvec==0: nvec=1
        x, y, z = axis/nvec
        angle = np.dot(vec_init,vec)/l
        sina = np.sqrt(1-angle**2)

        rotation = np.array([
            [x**2*(1-angle)+angle,
             y*x*(1-angle)-z*sina,
             z*x*(1-angle)+y*sina],
            [x*y*(1-angle)+z*sina,
             y**2*(1-angle)+angle,
             z*y*(1-angle)-x*sina],
            [x*z*(1-angle)-y*sina,
             y*z*(1-angle)+x*sina,
             z**2*(1-angle)+angle]
            ]).transpose()

        size = np.array([[length/lattice.aspect_ratio[i], 0, 0],
                         [0,    l, 0],
                         [0,    0, length/lattice.aspect_ratio[i]]])

        trans = np.dot(size, rotation)

        trans = np.concatenate(
            (trans, xyz.reshape(1,3)),
            axis=0)

        trans = np.concatenate(
            (trans.transpose(), np.ones((1,4))),
            axis = 0)
        trans[3,:] = np.array([-.3,-1,-.5,51])
        sc.add_cylinder(transform = trans,
                        color=color)

    r, g, b, _ = color
    r,g,b = pu.adjust_lightness((r/255,g/255,b/255), .2)
    r *= 255
    g *= 255
    b *= 255    

    vis = app.mesh(sc, depth_test=True, lighting=True,
                   shine=.2, emit=.3, light_color=(int(r),int(g),int(b),255))
    pan.add(vis)

    if in_app:
        return fig, pan, app
    else:
        app.run()
        app.destroy()
        sc.destroy()
        pass
#%%% Data visualization

def colorlat(lattice, values: np.ndarray, cmap: str = 'jet',
             title: str = None, box = False, axis = True, boundary = False):
    """
    Show a lattice with varying parameters over the elements.

    Parameters
    ----------
    lattice : Lattice-like object
        Lattice to be shown
    values : (nelems,1) numpy.ndarray
        Values of the elems to show
    cmap : str, optional
        name of the Datoviz colormap used. Default is jet.
    title : str, optional
        Name of the variable observed.
    box : BOOL, optional
        if you want to display the domain. The default is False.
    axis : BOOL, optional
        if you want to add an axis system. The default is True.
    boundary : BOOL, optional
        if you want to display in green the nodes at the boundary. The default is False.
    color : RGBA-TUPLE, optional
        color of the segments, by default blood red.

    Returns
    -------
    None.

    """
    dim = lattice.dim
    nodes = deepcopy(lattice.nodes)
    elems = deepcopy(lattice.elems)
    size = lattice.sizedom[:lattice.dim]
    if len(size)>1: r=size[1]/size[0]
    else: r=1
    norm = np.sqrt(sum(size**2))/dim
    if len(size)==1 : norm = np.sqrt(dim*size**2)/dim
    nodes /= norm

    # Create the figure
    cols=values
    colors = dvz.cmap(cmap, cols, vmin=min(cols), vmax=max(cols))

    fig1, app = initfig(ny=r*4/3)
    panel = fig1.panel((0,0),(xplot, yplot))

    if dim==2:panel.panzoom()
    else: panel.arcball()

    ax1 = draw_edges(app, dim, size/norm, nodes, elems, colors)
    panel.add(ax1)
    plotAxis(dim, size/norm, app, panel)

    if title is None: title=''
    colorbar(app, fig1, values, title, cmap=cmap)

    if box:
        bplot = plotBox(dim, size/norm, app)
        panel.add(bplot)

    if axis:
        plotAxis(dim, size/norm, app, panel)

    app.run()
    app.destroy()

    del app, fig1, panel, ax1


def displacement(nodes_init: np.ndarray, nodes_end: np.ndarray,
                 cmap: str = 'plasma'):
    """
    Show the node displacement, with colors depending on the length of the step.

    Parameters
    ----------
    nodes_init : (Nnode, ndim) numpyp.ndarray
        Nodes position before displacement
    nodes_end : (Nnode, ndim) numpyp.ndarray
        Nodes position after displacement.
    cmap : str, optional
        Colormap to visualize the lengteh of the steps. The default is 'plasma'.

    Returns
    -------
    None.

    """
    dim = nodes_init.shape[1]
    size = np.ptp(nodes_init, axis=0)
    r=size[1]/size[0]

    fig, app = initfig(ny=4/3*r)
    panel = fig.panel((0,0),(xplot, yplot))

    if dim==2: panel.panzoom()
    else: panel.arcball()

    if dim==2:
        pt0 = np.concatenate(
            (nodes_init, np.zeros((len(nodes_init),1))),
            axis=1
            )
        ptf = np.concatenate(
            (nodes_end, np.zeros((len(nodes_end),1))),
            axis=1
            )
    else:
        pt0 = deepcopy(nodes_init)
        ptf = deepcopy(nodes_end)

    size =  np.max(ptf, axis=0) - np.min(ptf, axis=0)
    norm = np.sqrt(np.sum(
        (np.max(pt0, axis=0) - np.min(pt0, axis=0))**2)
        )/dim

    length = np.sqrt(np.sum((ptf-pt0)**2, axis=1))

    colors = dvz.cmap(cmap, length, vmin=min(length), vmax=max(length))

    seg = app.segment(initial=pt0/norm, terminal=ptf/norm, linewidth=5,
                      color=colors,
                      cap=('round','triangle_out'),
                      depth_test=True)

    panel.add(seg)
    colorbar(app, fig, length, 'Displacement (mm)', cmap=cmap)

    plotAxis(dim, size/norm, app, panel)

    app.run()
    app.destroy()
    del app, fig, panel, seg
    pass


def scatter_lat(lattice, positions: np.ndarray, values: np.ndarray = None,
             cmap: str = 'viridis', m = 'cross', label=''):
    """
    Show specific points on a lattice.

    Parameters
    ----------
    lattice : lattice-like object
        The lattice on which you want to show markers.
    positions : np.ndarray
        Positions of the markers.
    values : np.ndarray, optional
        Values associated to the positions. The default is None.
    cmap : str, optional
        colormap to be used. The default is 'viridis'.
    m : str, otional
        Datoviz marker to use. Default is 'cross'.
        
    Notes
    -----
    See more on the markers at the `Datoviz documentation`_.
    
    .. _Datoviz documentation: https://datoviz.org/visuals/marker/
    """
    dim = lattice.dim
    size = lattice.sizedom[:lattice.dim]
    norm = np.sqrt(sum(size**2))/dim
    if len(size)==1 : norm = np.sqrt(dim*size**2)/dim
    if len(size)>2 and size[2]==0: dim = 2
    if len(size)>1: r=size[1]/size[0]
    else: r=1
    # Create the figure
    if values is None:
        fig, app = initfig(ny=r)
        panel = fig.panel()
    else:
        fig, app = initfig(ny=r*4/3)
        panel = fig.panel((0,0),(xplot, yplot))

    
    if dim==2:
        panel.panzoom()
    else:
        panel.arcball()

    # Plot the base network
    nw = app.segment(
        initial=lattice.nodes[lattice.elems[:,0].astype(int)]/norm,
        terminal=lattice.nodes[lattice.elems[:,1].astype(int)]/norm,
        color=[(62, 74, 131, 50)]*lattice.elems.shape[0],
        linewidth=3, depth_test=False)
    panel.add(nw)

    if values is not None:
        cols=values
        colorbar(app, fig, values, label, cmap=cmap)
    else: cols = np.zeros(positions.shape[0])
    colors = dvz.cmap(cmap, cols, vmin=min(cols), vmax=max(cols))

    # Add crosses on the given positions
    elm = app.marker(positions/norm, shape = m,
                     color = colors, size = [20]*positions.shape[0],
                     depth_test = True)
    panel.add(elm)

    plotAxis(dim, size/norm, app, panel)

    app.run()
    app.destroy()

    del app, fig, panel, nw, elm

#TODO: slicing
#TODO: element history
def slices(lattice):
    pass


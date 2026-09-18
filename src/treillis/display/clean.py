"""
Clean display
=============

Paper quality non-interactive plots.
"""

import os

import numpy as np                            
import matplotlib.pyplot as plt               
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.colors as mc

FONTPATH = [
    os.path.abspath(os.path.dirname(__file__) + r'/fonts/')
    ]
font_files = font_manager.findSystemFonts(fontpaths=FONTPATH)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

titlesfont = {'fontname':'Oranienbaum'}
textfont = {'fontname': 'FreeSans'}
mpl.style.use(os.path.abspath(os.path.dirname(__file__) + r'/PYLAT.mplstyle'))
mpl.rcParams['font.family'] = 'FreeSans'


from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.collections import LineCollection, PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

from . import plot_utilitaries as pu

RED = '#E50019'
DARK_BLUE = '#3E4A83'
DARK_GREY = '#1E1E1E'
LIGHT_BLUE = '#7E9CBB'
YELLOW = '#FFCD31'
MACARON = '#DA837B'
ARCHIPEL = '#00939D'
OPERA = '#BD987A'
WISTERIA = '#A72587'

mapmac = pu.make_cmap(['#F8E6E5', '#F0CECA', '#E3A39C', '#DA837B', '#AC5C56', 
                       '#8B3E3A'])
maparc = pu.make_cmap(['#CCE9EB', '#99d4d8', '#40aeb6', '#00939D', '#006e7a',
                       '#005561'])
mapope = pu.make_cmap(['#f2eae2', '#e5d5c6', '#ceb194', '#BD987A', '#91704a',
                       '#71532f'])
mapwis = pu.make_cmap(['#edd3e7', '#dca8cf', '#bd5ca5', '#A72587', '#89006d',
                       '#75005c'])

rainbow = [YELLOW, OPERA, MACARON, RED, WISTERIA, DARK_BLUE, LIGHT_BLUE, 
           ARCHIPEL]
rmap = pu.make_cmap(rainbow)

import treillis

xplot = 21/2.54/2
yplot = xplot

__all__ = ['showlat', 'shownodes', 'colorlat', 'displacement', 'scatter_lat',
           'tensor', 'animate']

#%% Useful functions
class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(axes3d.Axes3D, 'annotate3D', _annotate3D)


def signif(axe, data):
    """Create yticks for axe from data in scientific notation with a precision of 2."""
    # Find common order of magnitude to have most tick in format 1.*f
    data = np.linspace(min(data), max(data), num=7)
    exp = np.floor(np.log10(np.delete(abs(data), data==0)))
    exp = np.unique(exp)
    if len(exp)==1:
        oom = int(exp[0])
    elif len(exp)==2:
        oom = int(max(exp))
    elif len(exp)>2:
        oom = int(max(exp))-1
    elif len(exp)==0:
        oom=0

    if oom!=0:
        tick = ['%.2f'%(d/10**oom) for d in data]
        axe.set_ticks(data, labels=tick, fontsize=9)
        axe.ax.text(-1.5, 1.01,
                   r'$\times'+'10^{%s}'%oom+'$', transform=axe.ax.transAxes,
                   **textfont)
    else:
        tick = ['%.2f'%(d) for d in data]
        axe.set_ticks(data, labels=tick, fontsize=9)
        


def plotBox(dim, size, ax):
    """Plot a *box* domain."""
    if len(size) == 1 and dim == 2:
        circle = plt.Circle((0, 0), size[0], color=DARK_GREY, fill=False,
                            zorder=1000)
        ax.add_patch(circle)

    elif len(size) == 1 and dim == 3:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = size[0] * np.cos(u)*np.sin(v)
        y = size[0] * np.sin(u)*np.sin(v)
        z = size[0] * np.cos(v)
        ax.plot_wireframe(x, y, z, color=DARK_GREY, alpha=.3)
        ax.plot_surface(x, y, z, color=DARK_GREY, alpha=0.1)

    elif len(size) == 2:
        rect = plt.Rectangle((-size[0]/2, -size[1]/2), size[0], size[1],
                             color=DARK_GREY, fill=False, zorder=1000)
        ax.add_patch(rect)

    elif len(size) == 3:
        xx, yy = np.meshgrid(np.array([-size[0]/2, size[0]/2]),
                             np.array([-size[1]/2, size[1]/2]))
        ax.plot_wireframe(xx, yy, -size[2]/2 * np.ones(np.shape(xx)),
                          color=DARK_GREY, alpha=.3)
        ax.plot_surface(xx, yy, -size[2]/2 * np.ones(np.shape(xx)),
                        color=DARK_GREY, alpha=0.1)
        ax.plot_wireframe(xx, yy, size[2]/2 * np.ones(np.shape(xx)), 
                          alpha=.3, color=DARK_GREY)
        ax.plot_surface(xx, yy, size[2]/2 * np.ones(np.shape(xx)),
                        color=DARK_GREY, alpha=0.1)

        yy, zz = np.meshgrid(np.array([-size[1]/2, size[1]/2]),
                             np.array([-size[2]/2, size[2]/2]))
        ax.plot_wireframe(-size[0]/2 * np.ones(np.shape(xx)), yy, zz, 
                          alpha=.3, color=DARK_GREY)
        ax.plot_surface(-size[0]/2 * np.ones(np.shape(xx)), yy, zz, color=DARK_GREY,
                        alpha=0.1)
        ax.plot_wireframe(size[0]/2 * np.ones(np.shape(xx)), yy, zz, 
                          alpha=.3, color=DARK_GREY)
        ax.plot_surface(size[0]/2 * np.ones(np.shape(xx)), yy, zz, color=DARK_GREY,
                        alpha=0.1)

        xx, zz = np.meshgrid(np.array([-size[0]/2, size[0]/2]),
                             np.array([-size[2]/2, size[2]/2]))
        ax.plot_wireframe(xx, -size[1]/2 * np.ones(np.shape(xx)), zz, alpha=.3,
                          color=DARK_GREY)
        ax.plot_surface(xx, -size[1]/2 * np.ones(np.shape(xx)), zz, color=DARK_GREY,
                        alpha=0.1)
        ax.plot_wireframe(xx, -size[1]/2 * np.ones(np.shape(xx)), zz, 
                          alpha=.3, color=DARK_GREY)
        ax.plot_surface(xx, -size[1]/2 * np.ones(np.shape(xx)), zz, color=DARK_GREY,
                        alpha=0.1)

    elif len(size) == 5 and dim == 2:
        xy = np.array([[-size[0]/2, -size[1]/2], [size[0]/2, -size[1]/2],
                       [size[0]/2, size[1]/2], [-size[0]/2, size[1]/2],
                       [-size[0]/2, size[3]/2], [-size[0]/2 + size[4],
                                                 size[3]/2],
                       [-size[0]/2 + size[4], -size[3]/2], [-size[0]/2,
                                                            -size[3]/2]])
        wedge = plt.Polygon(xy, closed=True, color=DARK_GREY, fill=False)
        ax.add_patch(wedge)

    elif len(size) == 5 and dim == 3:
        x = [-size[0]/2, size[0]/2, size[0]/2, -size[0]/2, -size[0]/2,
             -size[0]/2 + size[4], -size[0]/2 + size[4], -size[0]/2]
        y = [-size[1]/2, -size[1]/2, size[1]/2, size[1]/2, size[3]/2,
             size[3]/2, -size[3]/2, -size[3]/2]
        z = [-size[2]/2] * 8
        verts = [list(zip(x,y,z))]
        poly = Poly3DCollection(verts, facecolors=DARK_GREY, alpha=0.1)
        ax.add_collection3d(poly)
        for j in range(7):
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], [z[j], z[j+1]], '-',
                    linewidth=1.5, color=DARK_GREY, alpha=.3)
        ax.plot([x[-1], x[0]], [y[-1], y[0]], [z[-1], z[0]], '-',
                linewidth=1.5, color=DARK_GREY, alpha=.3)
        z = [size[2]/2] * 8
        verts = [list(zip(x,y,z))]
        poly = Poly3DCollection(verts, facecolors=DARK_GREY, alpha=0.1)
        ax.add_collection3d(poly)
        for j in range(7):
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], [z[j], z[j+1]], '-',
                    linewidth=1.5, color=DARK_GREY, alpha=.3)
        ax.plot([x[-1], x[0]], [y[-1], y[0]], [z[-1], z[0]], '-',
                linewidth=1.5, color=DARK_GREY, alpha=.3)

        x = [[x[0], x[1]], [x[1], x[2]], [x[2], x[3]], [x[3], x[4]],
             [x[4], x[5]], [x[5], x[6]], [x[6], x[7]], [x[-1], x[0]]]
        y = [[y[0], y[1]], [y[1], y[2]], [y[2], y[3]], [y[3], y[4]],
             [y[4], y[5]], [y[5], y[6]], [y[6], y[7]], [y[-1], y[0]]]
        z = [-size[2]/2, size[2]/2]
        for i in range(len(x)):
            X = [x[i][0], x[i][1], x[i][1], x[i][0]]
            Y = [y[i][0], y[i][1], y[i][1], y[i][0]]
            Z = [z[0], z[0], z[1], z[1]]
            verts = [list(zip(X, Y, Z))]
            poly = Poly3DCollection(verts, facecolors=DARK_GREY, alpha=0.1)
            ax.add_collection3d(poly)
            for j in range(3):
                ax.plot([X[j], X[j+1]], [Y[j], Y[j+1]], [Z[j], Z[j+1]], '-',
                        linewidth=1.5,color=DARK_GREY)
            ax.plot([X[-1], X[0]], [Y[-1], Y[0]], [Z[-1], Z[0]], '-',
                    linewidth=1.5, color=DARK_GREY, alpha=.3)

    else:
        raise ValueError('Other kind of domains are not implemented. Feel free to add yours ;)')


def plotAxis(dim, size, ax):
    """Create an axis plot."""
    if dim == 2:
        if len(size) == 1:
            delta = 0.1 * size[0] * 2
        else:
            delta = max(0.1 * size[0], 0.1 * size[1])

        lim_x, lim_y = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(lim_x[0] - delta, lim_x[1])
        ax.set_ylim(lim_y[0] - delta, lim_y[1])
        X = lim_x[0]
        Y = lim_y[0]


        ax.arrow(X - 2 * delta/3, Y - 2 * delta/3, delta, 0,
                 width=0.5*delta/14, head_width=4.*delta/14,
                 head_length=4.*delta/14, color='g')
        ax.arrow(X - 2 * delta/3, Y - 2 * delta/3, 0, delta, width=0.5*delta/14,
                 head_width=4.*delta/14, head_length=4.*delta/14, color='g')
        ax.text(X - 1.5 * delta, Y - 0.2 * delta, 'y', color='k')
        ax.text(X - 0.2 * delta, Y - 1.5 * delta, 'x', color='k')

    else:
        if len(size) == 1:
            delta = 0.1 * size[0] * 2
        else:
            delta = max(0.1 * size[0], 0.1 * size[1], 0.1 * size[2])

        lim_x, lim_y, lim_z = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.set_xlim(lim_x[0] - delta, lim_x[1])
        ax.set_ylim(lim_y[0] - delta, lim_y[1])
        ax.set_zlim(lim_z[0] - delta, lim_z[1])
        X = 0#ax.get_xlim()[0]
        Y = -5#ax.get_ylim()[0]
        Z = lim_z[0]

        ax.quiver(X + 2*delta/3, Y - 6*delta, Z + 2*delta/3, 1,0,0,
                  length=delta, arrow_length_ratio = 0.3, linewidth=1.5, 
                  color='g')
        ax.quiver(X + 2*delta/3, Y - 6*delta, Z + 2*delta/3, 0, 1, 0,
                  length=delta, arrow_length_ratio = 0.3, linewidth=1.5, 
                  color='g')
        ax.quiver(X + 2*delta/3, Y - 6*delta, Z + 2*delta/3, 0, 0, 1,
                  length=delta, arrow_length_ratio = 0.3, linewidth=1.5, color='g', zorder=1000)
        ax.text(X + 1.5 * delta, Y - 6.3*delta, Z + 0.2 * delta, 'x',
                color='k')
        ax.text(X + 0.2 * delta, Y - 5*delta, (Z + 0.2 * delta)*0.9,
                'y', color='k')
        ax.text(X + 0.2 * delta, Y - 6*delta, Z + 1.5 * delta, 'z',
                color='k')

def color_depth(color):
    clist = [pu.adjust_lightness(color, i) 
            for i in [1.5*.5, 1.25*.5, .5, .5*.5, .25*.5]]
    cmap = pu.make_cmap(clist)
    return cmap

#%% Plotting
#%%% All the same functions as in Fast
#%%%% Basic visualization

def showlat(lattice, box: bool = True, 
            axis: bool = True, boundary = False, color = RED):
    """
    Show the lattice structure and some basic information.

    Parameters
    ----------
    lattice : lattice-like object
        Lattice-like object to observe.
    box : bool, optional
        Show the domain containing the lattice. Default is True.
    axis : bool, optional
        Show the axis. The default is True.
    boundary : bool, optional
        Emphasize the ``fixed_nodes``. The default is False.
    color : optional
        Color of the lattice.

    Returns
    -------
    matplotlib figure

    """
    raster = lattice.elems.shape[0] > 500
    wdt = 1.5 * (lattice.L/lattice.sizedom.mean()*30)
    if lattice.dim == 2:
        if len(lattice.sizedom)>1: 
            yplot = lattice.sizedom[1]/lattice.sizedom[0] * xplot
        else: yplot = xplot
        fig, ax = plt.subplots(figsize=(xplot,yplot))
        ax.set_aspect('equal', adjustable='box')

        ax.plot(lattice.nodes[lattice.elems[:,:2].astype(int),0].T,
                lattice.nodes[lattice.elems[:,:2].astype(int),1].T,
                color = color, linewidth=wdt, rasterized=raster)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        eps = max(lattice.length)
        
        if boundary:
            ax.plot(lattice.nodes[lattice.fixed_nodes, 0],
                    lattice.nodes[lattice.fixed_nodes, 1],
                    'o', color=ARCHIPEL, 
                    markersize=4* (lattice.L/lattice.sizedom.mean()*30))
        
        ax.set_xlim(left=min(lattice.nodes[:,0]) - eps,
                    right=max(lattice.nodes[:,0])  + eps)
        ax.set_ylim(bottom=min(lattice.nodes[:,1]) - eps,
                    top=max(lattice.nodes[:,1]) + eps)
            
        ax.spines[['top','bottom','left','right']].set_visible(False)
        
    else:
        if len(lattice.sizedom)>1: 
            yplot = lattice.sizedom[1]/lattice.sizedom[0] * xplot
        else: yplot = xplot
        fig, ax = plt.subplots(figsize=(xplot,yplot),
                               subplot_kw={"projection": "3d"})
        segs = np.concatenate((
            lattice.nodes[lattice.elems[:,0].astype(int)]\
                .reshape(lattice.elems.shape[0], 1, lattice.dim),
            lattice.nodes[lattice.elems[:,1].astype(int)]\
                .reshape(lattice.elems.shape[0], 1, lattice.dim)),
                axis=1)
            
        D = np.sqrt(np.sum(
            (lattice.nodes[lattice.elems[:,0].astype(int)]
             + lattice.nodes[lattice.elems[:,1].astype(int)])**2 / 4,
            axis=1))
        norm = mc.Normalize(vmin=0, vmax=1.5*lattice.sizedom[0])
        cmap = color_depth(color)
        lines = Line3DCollection(segs, linewidths=wdt/2, 
                                 colors = cmap(norm(D)), 
                                 rasterized=raster)
        
        ax.add_collection(lines)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(azim=-60, elev=35)
        ax.set_axis_off()
        eps = max(lattice.length)
        
        if boundary:
            ax.plot(lattice.nodes[lattice.fixed_nodes, 0],
                    lattice.nodes[lattice.fixed_nodes, 1],
                    lattice.nodes[lattice.fixed_nodes, 2],
                    'o', alpha=.3, color=ARCHIPEL, 
                    markersize=4* (lattice.L/lattice.sizedom.mean()*30), 
                    zorder=1000)
        
        
        ax.set_xlim(left=min(lattice.nodes[:,0]) - eps/2,
                    right=max(lattice.nodes[:,0]) + eps/2)
        ax.set_ylim(bottom=min(lattice.nodes[:,1]) - eps/2,
                    top=max(lattice.nodes[:,1]) + eps/2)
        ax.set_zlim(bottom=min(lattice.nodes[:,2]) - eps/2,
                    top=max(lattice.nodes[:,2]) + eps/2)
    
    if box:
        plotBox(lattice.dim, lattice.sizedom, ax)

    if axis:
        plotAxis(lattice.dim, lattice.sizedom, ax)
    
    
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig


def shownodes(lattice, arrows: np.ndarray = None, box: bool = True, 
            axis: bool = True, boundary = False, color = RED):
    """
    Show the lattice nodes and some basic information.

    Parameters
    ----------
    lattice : lattice-like object
        Lattice-like object to observe.
    arrows : np.ndarray, optional
        Arrows indicating the displacement applied to the ``fixed_nodes``.
    box : bool, optional
        Show the domain containing the lattice. Default is True.
    axis : bool, optional
        Show the axis. The default is True.
    boundary : bool, optional
        Emphasize the ``fixed_nodes``. The default is False.
    color : optional
        Color of the lattice.

    Returns
    -------
    matplotlib figure

    """
    raster = lattice.nodes.shape[0] > 1000
    if lattice.dim == 2:
        if len(lattice.sizedom)>1: 
            yplot = lattice.sizedom[1]/lattice.sizedom[0] * xplot
        else: yplot = xplot
        fig, ax = plt.subplots(figsize=(xplot,yplot))
        ax.set_aspect('equal', adjustable='box')

        ax.plot(lattice.nodes[:,0],
                lattice.nodes[:,1],'o', 
                color = color, rasterized=raster,
                markersize=4*(lattice.L/lattice.sizedom.mean()*30))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        eps = max(lattice.length)
        
        if boundary:
            ax.plot(lattice.nodes[lattice.fixed_nodes, 0],
                    lattice.nodes[lattice.fixed_nodes, 1],
                    'o', color=ARCHIPEL, markersize=4* (lattice.L/lattice.sizedom.mean()*30))
        
        if arrows is not None:
            ax.quiver(lattice.nodes[lattice.fixed_nodes,0],
                      lattice.nodes[lattice.fixed_nodes,1],
                        arrows[:,0], arrows[:,1], pivot='tip',
                        rasterized=len(lattice.fixed_nodes)>100,
                        color=DARK_BLUE)
            
            arrowlength = np.sqrt(np.sum(arrows**2, axis=1)).max()
            
            ax.set_xlim(left=min(lattice.nodes[:,0]) - arrowlength - eps,
                        right=max(lattice.nodes[:,0]) + arrowlength + eps)
            ax.set_ylim(bottom=min(lattice.nodes[:,1]) - arrowlength - eps,
                        top=max(lattice.nodes[:,1]) + arrowlength + eps)
        else:
            ax.set_xlim(left=min(lattice.nodes[:,0]) - eps,
                        right=max(lattice.nodes[:,0])  + eps)
            ax.set_ylim(bottom=min(lattice.nodes[:,1]) - eps,
                        top=max(lattice.nodes[:,1]) + eps)
            
        ax.spines[['top','bottom','left','right']].set_visible(False)
        
    else:
        if len(lattice.sizedom)>1: 
            yplot = lattice.sizedom[1]/lattice.sizedom[0] * xplot
        else: yplot = xplot
        fig, ax = plt.subplots(figsize=(xplot,yplot),
                               subplot_kw={"projection": "3d"})
        
        ax.plot(lattice.nodes[:,0],
              lattice.nodes[:,1],
              lattice.nodes[:,2], 'o', markersize=3,
              color=RED, alpha=.6, rasterized=raster)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect('equal', adjustable='box')
        ax.view_init(azim=-60, elev=35)
        ax.set_axis_off()
        eps = max(lattice.length)
        
        if boundary:
            ax.plot(lattice.nodes[lattice.fixed_nodes, 0],
                    lattice.nodes[lattice.fixed_nodes, 1],
                    lattice.nodes[lattice.fixed_nodes, 2],
                    'o', alpha=.3, color=ARCHIPEL, markersize=4, zorder=1000)
        
        if arrows is not None:
            ax.quiver(lattice.nodes[lattice.fixed_nodes,0],
                      lattice.nodes[lattice.fixed_nodes,1],
                      lattice.nodes[lattice.fixed_nodes,2],
                        arrows[:,0], arrows[:,1], arrows[:,2], pivot='tip',
                        rasterized=len(lattice.fixed_nodes)>100,
                        color=DARK_BLUE)
            
            arrowlength = np.sqrt(np.sum(arrows**2, axis=1)).max()
            
            ax.set_xlim(left=min(lattice.nodes[:,0]) - arrowlength - eps,
                        right=max(lattice.nodes[:,0]) + arrowlength + eps)
            ax.set_ylim(bottom=min(lattice.nodes[:,1]) - arrowlength - eps,
                        top=max(lattice.nodes[:,1]) + arrowlength + eps)
            ax.set_zlim(bottom=min(lattice.nodes[:,2]) - arrowlength - eps,
                        top=max(lattice.nodes[:,2]) + arrowlength + eps)
            
        else:
            ax.set_xlim(left=min(lattice.nodes[:,0]) - eps/2,
                        right=max(lattice.nodes[:,0]) + eps/2)
            ax.set_ylim(bottom=min(lattice.nodes[:,1]) - eps/2,
                        top=max(lattice.nodes[:,1]) + eps/2)
            ax.set_zlim(bottom=min(lattice.nodes[:,2]) - eps/2,
                        top=max(lattice.nodes[:,2]) + eps/2)
    
    if box:
        plotBox(lattice.dim, lattice.sizedom, ax)

    if axis:
        plotAxis(lattice.dim, lattice.sizedom, ax)
    
    
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return fig




#%%%% Data visualization

def colorlat(lattice, values: np.ndarray, cmap: mc.Colormap = maparc, 
             nrows: int = 1, ncols: int = 1,
             label: str|list[str] = None, unified: bool = False,
             uni_label: str = None,
             box: bool = False, axis: bool = True):
    """
    Show values over the lattice elements.
    
    You can choose to have a unified colormap over all values to see them
    relative to one another.

    Parameters
    ----------
    lattice : lattice-like object
        Lattice to show.
    values : (Nelems, Nvalues) np.ndarray
        Nvalues that you want to show over the elements.
    cmap : mc.Colormap, optional
        Colormap to use.
    nrows : int, optional
        Number of rows in the figure to show several values. The default is 1.
    ncols : int, optional
        Number of columns in the figure to show several values. 
        The default is 1.
    label : str|list[str], optional
        Label of the individual axes.
    unified : bool, optional
        Have one unified colorbar for all values. The default is False.
    uni_label : str, optional
        Label for the unified colorbar.
    box : bool, optional
        Show the lattice domain. The default is False.
    axis : bool, optional
        Show the lattice axis. The default is True.

    Returns
    -------
    fig : matplotlib figure

    """
    raster = lattice.elems.shape[0] > 500
    if len(lattice.sizedom)>1: 
        yplot = lattice.sizedom[1]/lattice.sizedom[0] * xplot
    else: yplot = xplot
    
    if not isinstance(label, (list, np.ndarray)): label = [label]
        
    # Dealing with the shape of the figure if there are several values to show
    if len(values.shape) > 1:
        raster = lattice.elems.shape[0] * values.shape[1] > 500
        
        if nrows==1 and ncols==1:
            ncols = 3*(values.shape[1]>=3)\
                + values.shape[1]*(values.shape[1]<3)
            nrows = values.shape[1] // 3 + (values.shape[1]%3 != 0)
        if nrows * ncols < values.shape[1]:
            raise Warning('Dimensions of the figure do not correspond to the amount of values shown, increasing the size to plot everything.')
            
            if ncols < nrows:
                ncols = values.shape[1] // nrows + (values.shape[1]%nrows != 0)
            else:
                nrows = values.shape[1] // ncols + (values.shape[1]%ncols != 0)
    else: values = values.reshape(values.shape[0], 1) 
    
    # Drawing the lattice
    if lattice.dim == 3: 
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                               figsize=(xplot * ncols, yplot * nrows),
                               subplot_kw={"projection": "3d"})
    else: 
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                               figsize=(xplot * ncols, yplot * nrows))
    
    eps = max(lattice.length)
    for i in range(values.shape[1]):
        if nrows>1 and ncols>1: axe = ax[i//nrows, i%ncols]
        elif values.shape[1]>1: axe = ax[i]
        else: axe=ax
        
        if unified: norm = mc.Normalize(vmin=values.min(), vmax=values.max())
        else: norm = mc.Normalize(vmin=values[:,i].min(), vmax=values[:,i].max())
        
        segs = np.concatenate((
            lattice.nodes[lattice.elems[:,0].astype(int)]\
                .reshape(lattice.elems.shape[0], 1, lattice.dim),
            lattice.nodes[lattice.elems[:,1].astype(int)]\
                .reshape(lattice.elems.shape[0], 1, lattice.dim)),
                axis=1)
            
        if lattice.dim == 3:
            lines = Line3DCollection(segs, linewidths=1.5* (lattice.L/lattice.sizedom.mean()*60), 
                                     colors = cmap(norm(values[:,i])), 
                                     rasterized=raster)
            axe.set_zlim(bottom=min(lattice.nodes[:,2]) - eps/2,
                        top=max(lattice.nodes[:,2]) + eps/2)
            axe.view_init(azim=-60, elev=35)
            axe.set_zticks([])
            # axe.set_axis_off()
        else:
            lines = LineCollection(segs, linewidths=1.5* (lattice.L/lattice.sizedom.mean()*60), 
                                     colors = cmap(norm(values[:,i])), 
                                     rasterized=raster)
            # axe.spines[['top','bottom','left','right']].set_visible(False)
        
        axe.add_collection(lines)
        axe.set_xlim(left=min(lattice.nodes[:,0]) - eps/2,
                    right=max(lattice.nodes[:,0]) + eps/2)
        axe.set_ylim(bottom=min(lattice.nodes[:,1]) - eps/2,
                    top=max(lattice.nodes[:,1]) + eps/2)
        axe.set_aspect('equal', adjustable='box')
        axe.set_xticks([])
        axe.set_yticks([])
        
        if unified and values.shape[1]>1:
            if i < len(label):
                axe.set_title(label[i], fontsize=20, **titlesfont)
        else:
            mappable = mpl.cm.ScalarMappable(norm = norm, cmap=cmap)
            if lattice.dim==2:
                divider = make_axes_locatable(axe)
                width = axes_size.AxesY(axe, aspect=1./20)
                pad = axes_size.Fraction(.5, width)
                cax = divider.append_axes(position="right", size=width, pad=pad)
                cb = fig.colorbar(mappable, cax=cax)
            else:
                cb = fig.colorbar(mappable, ax=axe, fraction=.03)
            signif(cb, values[:,i])
            
            if i < len(label): 
                cb.set_label(label[i], fontsize=18, **titlesfont)
    
        if box:
            plotBox(lattice.dim, lattice.sizedom, axe)
    
        if axis:
            plotAxis(lattice.dim, lattice.sizedom, axe)
            
    if unified and values.shape[1]>1:
        mappable = mpl.cm.ScalarMappable(norm = norm, cmap=cmap)
        cax = fig.add_axes([.9, .1, .02, .8])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(uni_label, fontsize=18, **titlesfont)
        
    # Delete empty axes
    for i in range(ncols * nrows):
        if nrows>1 and ncols>1: axe = ax[i//nrows, i%ncols]
        elif values.shape[1]>1: axe = ax[i]
        else: axe=ax
        if not axe.has_data(): plt.delaxes(ax=axe)

    return fig


def displacement(nodes_init: np.ndarray, displ: np.ndarray,
                 values: np.ndarray = None, cmap: mc.Colormap = mapmac,
                 label: str = r'Displacement [mm]'):
    """
    Plot the nodes displacement

    Parameters
    ----------
    nodes_init : (Nnodes, Ndim) np.ndarray
        Initial position of the nodes.
    displ : (Nnodes, Ndim) np.ndarray
        Displacement.
    values : (Nnodes,) np.ndarray, optional
        Additional value giving color to the displacement field.
        The default is the length of displacement.
    cmap : mc.Colormap, optional
        Colormap to use for values.
    label : str, optional
        colormap label. The default is r'Displacement [mm]'.

    Returns
    -------
    fig : matplotlib figure.

    """
    dim = nodes_init.shape[1]
    
    if values is None:
        values = np.sqrt(np.sum(displ**2, axis=1))
    norm = mc.Normalize(vmin=values.min(), vmax=values.max())
    if (values.max()-values.min())/values.max()<.05:
        norm = mc.Normalize(vmin=values.min()*.5, vmax=values.max()*1.5)
    mappable = mpl.cm.ScalarMappable(norm = norm, cmap=cmap)
        
    if dim == 3: 
        fig, ax = plt.subplots(figsize=(xplot, yplot),
                               subplot_kw={"projection": "3d"})
        ax.quiver(nodes_init[:,0],
                  nodes_init[:,1],
                  nodes_init[:,2],
                  displ[:,0], displ[:,1], displ[:,2],
                  color = cmap(norm(values)), scale=1, scale_units='xy')
        
        ax.view_init(azim=-60, elev=35)
        ax.set_zticks([])
        cb = fig.colorbar(mappable, ax=ax, fraction=.03)
    
        
    else: 
        fig, ax = plt.subplots(figsize=(xplot, yplot))
        ax.quiver(nodes_init[:,0],
                  nodes_init[:,1],
                  displ[:,0], displ[:,1], 
                  color = cmap(norm(values)), scale=1, scale_units='xy')
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1./20)
        pad = axes_size.Fraction(.5, width)
        cax = divider.append_axes(position="right", size=width, pad=pad)
        cb = fig.colorbar(mappable, cax=cax)
        
    signif(cb, values)
    cb.set_label(label, fontsize=18, **titlesfont)
        
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([])
    ax.set_yticks([])
    
    Mn = np.max(nodes_init, axis=0)
    mn = np.min(nodes_init, axis=0)

    
    plotAxis(dim, Mn-mn, ax)
    return fig

def scatter_lat(lattice, positions: np.ndarray, values: np.ndarray = None,
                cmap: mc.Colormap = mapwis, marker: str = 'x', 
                label: str = '', color=DARK_BLUE):
    """
    Plot a scatter of points over the lattice.

    Parameters
    ----------
    lattice : lattice-like object
        Lattice to plot.
    positions : (Npoints, Ndim) np.ndarray
        Positions of the scattered points.
    values : (Npoints,) np.ndarray, optional
        Values associated to the scattered points.
    cmap : mc.Colormap, optional
        Colormap for the scattered points.
    marker : str, optional
        Marker shape. The default is 'x'.
    label : str, optional
        Label of the scattered points.

    Returns
    -------
    fig : matplotlib figure

    """
    col = mc.to_rgba(color, alpha=.2)
    fig = showlat(lattice, box=False, color=col)
    nocb = False
    if values is None:
        values = np.zeros(positions.shape[0])
        nocb = True
    norm = mc.Normalize(vmin=values.min(), vmax=values.max())
    if (values.max()-values.min())/values.max()<.05:
        norm = mc.Normalize(vmin=values.min()*.5, vmax=values.max()*1.5)
    mappable = mpl.cm.ScalarMappable(norm = norm, cmap=cmap)
    
    ax = fig.get_axes()[0]
    if lattice.dim==3:
        ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                   c=values, cmap=cmap, marker=marker)
        ax.view_init(azim=-60, elev=35)
        ax.set_zticks([])
        if not nocb:
            cb = fig.colorbar(mappable, ax=ax, fraction=.03)
        
    if lattice.dim==2:
        ax.scatter(positions[:,0], positions[:,1],
                   c=values, cmap=cmap, marker=marker)
        if not nocb:
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1./20)
            pad = axes_size.Fraction(.5, width)
            cax = divider.append_axes(position="right", size=width, pad=pad)
            cb = fig.colorbar(mappable, cax=cax)
        
    if not nocb:
        signif(cb, values)
        cb.set_label(label, fontsize=18, **titlesfont)
    if nocb:
        ax.set_title(label, fontsize=20, **titlesfont)
        
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig



#%%% Something new
def tensor():
    pass

def animate():
    pass
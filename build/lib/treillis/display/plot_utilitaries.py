import colorsys as cs
from itertools import cycle
import numpy as np
import matplotlib.colors as mc

markers = ['o', 'v', '^', 's', 'D']
mark=cycle(markers)

line_list = ['-',':','--','-.',(0, (3, 3, 1, 3, 1, 3)), (0, (3, 1, 1, 1))]


MATLAB = [(0.00,0.45,0.74),
    (0.85,0.33,0.10),
    (0.93,0.69,0.13),
    (0.49,0.18,0.56),
    (0.47,0.67,0.19),
    (0.30,0.75,0.93),
    (0.64,0.08,0.18),
    (1, 1, 0.07)
   ]


def couleur(index, Len, palette):
    n = len(palette) - 1
    cmin = np.floor(index/Len*n)
    cmax = np.ceil(index/Len*n)
        
    if cmin != cmax:
        x = (index/Len*n - cmax) / (cmin - cmax)
        cmin = mc.to_rgb(palette[int(cmin)])
        cmax = mc.to_rgb(palette[int(cmax)])      
    
        return (x*cmin[0]+(1-x)*cmax[0], x*cmin[1]+(1-x)*cmax[1], x*cmin[2]+(1-x)*cmax[2])
    else:
        return palette[int(cmin)]

def adjust_lightness(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    return couleur(int(amount*100), 100, ['white', color, 'black'])

def color_marker(color, alpha=0.6):
    cola = mc.to_rgb(color)
    cola = [cola[0], cola[1], cola[2], alpha]
    return color, cola

def make_cmap(colorlist):
    listed = [mc.to_rgb(couleur(i, 256, colorlist)) for i in range(256)]
    return mc.ListedColormap(listed)


MLcmap = [mc.to_rgb(couleur(i, 256, MATLAB)) for i in range(256)]

   
def to_cmyk(color):
    r,g,b=mc.to_rgb(color)
    norm = max(r,g,b)
    k = 1-norm
    r/=norm
    g/=norm
    b/=norm
    
    c = (g+b)/2
    m = (r+b)/2
    y = (r+g)/2

    return c,m,y,k

def cmyk_to_rgb(color):
    c,m,y,k = color
    r = (1-k) * (m+y-c)
    g = (1-k) * (c+y-m)
    b = (1-k) * (c+m-y)
    S=r+g+b
    return np.round(r,5), np.round(g,5), np.round(b,5)

def make_gradient(col1,col2,n=10, mod='cmyk'):
    if mod=='cmyk':
        col1_cmyk = to_cmyk(col1)
        col2_cmyk = to_cmyk(col2)
        
        if col1_cmyk[3]<=col2_cmyk[3]:
            colm = [col1_cmyk[0],col1_cmyk[1],col1_cmyk[2],col1_cmyk[3]]
            colM = [col2_cmyk[0],col2_cmyk[1],col2_cmyk[2],col2_cmyk[3]]
        else:
            colm = [col2_cmyk[0],col2_cmyk[1],col2_cmyk[2],col2_cmyk[3]]
            colM = [col1_cmyk[0],col1_cmyk[1],col1_cmyk[2],col1_cmyk[3]]
            
        if colM[3]-colm[3]<.5 and colm[3]>.25 and colM[3]<.75:
            colm[3] = .25
            colM[3] = .75
        elif colM[3]-colm[3]<.5 and colm[3]<=.25: colM[3] = colm[3]+.5
        elif colM[3]-colm[3]<.5 and colM[3]>=.75: colm[3] = colM[3]-.5
    
        palette = [cmyk_to_rgb((colM[0]*x + colm[0]*(1-x),
                    colM[1]*x + colm[1]*(1-x),
                    colM[2]*x + colm[2]*(1-x),
                   colM[3]*x + colm[3]*(1-x))) for x in np.linspace(0,1,n)]
        return palette
    
    elif mod=='rgb':
        return [couleur(i,n,[col1,col2]) for i in range(n)]
    
    elif mod=='hsv':
        col1=mc.to_rgb(col1)
        col1_hsv = cs.rgb_to_hsv(col1[0],col1[1],col1[2])
        col1_hsv=[col1_hsv[0],col1_hsv[1],col1_hsv[2]]
        col2=mc.to_rgb(col2)
        col2_hsv = cs.rgb_to_hsv(col2[0],col2[1],col2[2])
        col2_hsv=[col2_hsv[0],col2_hsv[1],col2_hsv[2]]
        if col1_hsv[0]<=col2_hsv[0]:
            colm = [col1_hsv[0],col1_hsv[1],col1_hsv[2]]
            colM = [col2_hsv[0],col2_hsv[1],col2_hsv[2]]
        else:
            colm = [col2_hsv[0],col2_hsv[1],col2_hsv[2]]
            colM = [col1_hsv[0],col1_hsv[1],col1_hsv[2]]
            
        if abs(colM[2]-colm[2])<0.15:
            if colM[2]>colm[2]:
                m = colM[2]-colm[2]
                colm[2]+= (.15-m)*(colm[2]>=.15) - .15*(colM[2]>=.85)
                colM[2] += (.15-m)*(colM[2]<=.85) + .15*(colm[2]<=.15)
            else:
                m = colm[2]-colM[2]
                colm[2]+= (.15-m)*(colm[2]<=.85) + .15*(colM[2]<=.15)
                colM[2] += (.15-m)*(colm[2]>=.15) - .15*(colm[2]>=.85)
    
        if abs(colM[0]-colm[0])<=abs(colm[0]-colM[0]%-1):    
            palette = [cs.hsv_to_rgb(colM[0]*x + colm[0]*(1-x),
                        colM[1]*x + colm[1]*(1-x),
                        colM[2]*x + colm[2]*(1-x)) for x in np.linspace(0,1,n)]
        else:
            M=colM[0]%-1
            palette = [cs.hsv_to_rgb((colm[0] - (colm[0]-M)*x)%1,
                        colM[1]*x + colm[1]*(1-x),
                        colM[2]*x + colm[2]*(1-x)) for x in np.linspace(0,1,n)]
        return palette
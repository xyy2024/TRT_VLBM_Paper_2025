#!/usr/bin/python
# -*- coding: utf-8 -*-

#    _plottools.py
#    Copyright (C) 2025 Xu Yuyang (https://github.com/xyy2024)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 

'''# A tool module designed for using matplotlib easily

### About Animation Output

If you want to output animations, please install FFmpeg. 
Besides, I also recommend installing openh264.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 1.728*12
plt.rcParams["figure.labelsize"] = 24
plt.rcParams["figure.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['lines.markersize'] = 10

# Rank figure
from numpy.polynomial import Polynomial as poly
from math import isnan

# Animation figure
import os
import subprocess

# Log
import datetime as DT

# Typing
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numbers import Real
from typing import Callable, Sequence

__all__ = ["gridfig", "show", "Save", "save", "prt_2d", "prt_mask_2d", "rankfig", "rank_save"]

# Init, Show and Save

def gridfig(
        rows:int = 1, cols:int = 1, 
        sharex:str = "all", sharey:str = "all",
        suptitle:str = "", supxlabel:str = "", supylabel:str = ""
        ) -> tuple[Figure, Sequence[Axes]]:
    ''' plt.subplots and something else '''
    fig:Figure
    fig, _axs = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, squeeze=False)
    fig.set_figheight(rows*8)
    fig.set_figwidth(cols*8)
    if suptitle != "": fig.suptitle(suptitle)
    if supxlabel != "": fig.supxlabel(supxlabel)
    if supylabel != "": fig.supylabel(supylabel)
    return fig, _axs.flatten()

def show(fig:Figure, show_fig:bool = True) -> None:
    ''' plt.show and something else '''
    if show_fig: plt.show()
    else: plt.close(fig)

class Save:
    '''### Used for automatically handling settings when storing figures.

    #### Basic Usage：
    >>> save = Save()
    >>> save(fig)

    For an instance "save" of this class, the image will be saved to:

    * `f'{save.dir}/{save.prefix}_{self.file_id:0=5}.png'`
    
    which `{self.file_id:0=5}` represents a five-digit auto-incrementing decimal number, the number
    is used to avoid OSError when the file already exist and been locked or setted to read-only.
    
    You can set these attributes by

    >>> save = Save('test001', 'test_graphic')

    >>> save = Save(prefix = 'test001', dir = 'test_graphic')

    >>> save = Save()
    >>> save.prefix = 'test001'
    >>> save.dir = 'test_graphic'

    Caution: every time you changed these attributes, the "file_id" will be reset to zero.

    #### Saving animation
    After saving a bunch of figures, you can combine these image with method "Save.animation()"
    
    For detailed information, please refer to the documentation of Save.animation(). 
    This feature relies on FFmpeg (required) and openh264 (optional).
    '''
    file_id = 0

    def __init__(self, prefix:str = "test_result", dir:str = "graphic"):
        self._prefix = prefix
        self._dir = os.path.abspath(dir)
        self.cwd = os.getcwd()

    @property
    def prefix(self):
        return self._prefix
    @prefix.setter
    def prefix(self, prefix):
        self._prefix = prefix
        self.file_id = 0
    
    @property
    def dir(self):
        return self._dir
    @dir.setter
    def dir(self, dir):
        self._dir = dir
        self.file_id = 0

    def __call__(self, fig:Figure, save_fig:bool = True):
        if not save_fig:
            return None
        
        if os.path.exists(self.dir):
            if not os.path.isdir(self.dir):
                raise FileExistsError(f"Cannot make directory: {self.dir!r}")
        else:
            os.makedirs(self.dir)
        
        os.chdir(self.dir)
        while True:
            try:
                fig.savefig(f"{self.prefix}_{self.file_id:0=5}.png")
                break
            except:
                self.file_id += 1
        os.chdir(self.cwd)
        self.file_id += 1
        return None

    def animation(self, fps:float, file_name:str = 'output.mp4'):
        '''### Combine the previously stored images into an animation. 

        #### Parameters
        * `fps`: number of images per second
        * `file_name`：output file name, default is 'output.mp4' 

        The animation will be stored in the same folder as the image.
        '''
        if self.file_id == 0: return None
        os.chdir(self.dir)
        file_id = self.file_id
        while True: # This simply works
            try:
                os.remove(f"{self.prefix}_{file_id:0=5}.png")
                file_id += 1
            except FileNotFoundError:
                break
        cmd = [
            'ffmpeg',
            '-y',                            # Overwrite output files without warning
            '-framerate', str(fps),
            '-i', f'{self.prefix}_%05d.png', # Input from files
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'fatal',            # Default output is too many
            file_name]
        subprocess.run(cmd, check=True)
        os.chdir(self.cwd)

save = Save()

# Log tool
def get_timestamp():
    return DT.datetime.today().isoformat('T', timespec='microseconds').replace(":", "_").replace("-", "_")

# Flow Fig
cmap_positive_ignorezero = LinearSegmentedColormap.from_list(
    name = 'plottools_cmap_positive_ignorezero', 
    colors = [(1, 1, 1), (0.25, 0.75, 0.75), (0, 0, 0.5)], 
    N = 10000)
cmap_positive_focuszero = "copper"
cmap_ignorezero = "bwr"
cmap_focuszero = LinearSegmentedColormap.from_list(
    name = 'plottools_cmap_focuszero', 
    colors = [(0.5, 0.8, 1), (0, 0.5, 1), (0, 0, 0), (1, 0.5, 0), (1, 0.8, 0.5)], 
    N = 10000)
cmap = "coolwarm"
unbroken_streamline = dict(broken_streamlines = False, linewidth = 0.3, density = (0.75, 0.75), arrowsize=2)

def prt_2d(
    x:np.ndarray, y:np.ndarray, 
    u:np.ndarray, v:np.ndarray|None = None, 
    fig:Figure|None = None, ax:Axes|None = None, 
    xlabel:str="", ylabel:str="", title:str="",
    cmap:str|Colormap=cmap_positive_ignorezero, show:bool = False, log:bool = False,
    **kwargs
    ) -> None:
    '''Draw 2d fig in given axes.

    I use `u[i, j]` to store $u(x_i, y_j)$ instead of `u[j, i]` (which is matplotlib's default).
    So I need a function to automatically transfer `u` to `u.T` to prevent mistakes.

    ### Parameters:
    * `x`: `numpy.ndarray`. Same as the parameter `x` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `y`: `numpy.ndarray`. Same as the parameter `y` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `u`: `numpy.ndarray`. 
    * `v`: `numpy.ndarray|None`. If the parameter `v` is `None`, this function will draw a contourf plot at given 
    figure and axes. If the parameter `v` is a `ndarray`, this function will draw a 
    streamplot at given figure and axes.
    * `fig`: `matplotlib.figure.Figure|None`. If `fig` is `None`, this function will create a new figure and axes.
    * `ax`: `matplotlib.axes.Axes|None`. If `ax` is `None`, this function will create a new figure and axes.
    * `xlabel`, `ylabel`, `title`: `str`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `cmap`: `str|matplotlib.colors.Colormap`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `show`: `bool`. If the parameter is `True`, the function will call `matplotlib.pyplot.show()`. Default is `False`.
    * `log`: `bool`. If the parameter is `True`, the function will save the value of these parameters: `x`,`y`,`u`,`v`,`xlabel`,`ylabel`,`title`
    to a file in the 'log' folder.
    * `**kwargs`: other parameters that could pass to `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    '''
    if v is None:
        return prt_dens_2d(x, y, u, fig, ax, xlabel, ylabel, title, cmap, show, log, **kwargs)
    else:
        return prt_flow_2d(x, y, u, v, fig, ax, xlabel, ylabel, title, cmap, show, log, **kwargs)

def prt_flow_2d(
    x:np.ndarray, y:np.ndarray, 
    u:np.ndarray, v:np.ndarray, 
    fig:Figure|None = None, ax:Axes|None = None, 
    xlabel:str="", ylabel:str="", title:str="",
    cmap:str|Colormap=cmap_positive_ignorezero, show:bool = False, log:bool = False,
    **kwargs
    ):
    '''Draw streamplot in given axes.
    
    ### Parameters:
    * `x`: `numpy.ndarray`. Same as the parameter `x` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `y`: `numpy.ndarray`. Same as the parameter `y` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `u`: `numpy.ndarray`. 
    * `v`: `numpy.ndarray`. 
    * `fig`: `matplotlib.figure.Figure|None`. If `fig` is `None`, this function will create a new figure and axes.
    * `ax`: `matplotlib.axes.Axes|None`. If `ax` is `None`, this function will create a new figure and axes.
    * `xlabel`, `ylabel`, `title`: `str`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `cmap`: `str|matplotlib.colors.Colormap`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `show`: `bool`. If the parameter is `True`, the function will call `matplotlib.pyplot.show()`. Default is `False`.
    * `log`: `bool`. If the parameter is `True`, the function will save the value of these parameters: `x`,`y`,`u`,`v`,`xlabel`,`ylabel`,`title`
    to a file in the 'log' folder.
    * `**kwargs`: other parameters that could pass to `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    '''
    if log:
        if os.path.exists('log'):
            if os.path.isfile('log'):
                raise OSError('Cannot create directory `log` for storing logs.')
        else:
            os.mkdir('log')
        timestamp = get_timestamp()
        with open(f'log\\{timestamp}_Streamplot.py', mode='x') as file:
            file.write('#!/usr/bin/python\n'
                       '# -*- coding: utf-8 -*-\n\n'
                       'from _plottools import prt_flow_2d\n'
                       'import numpy as np\n\n'
                       'def prt(fig, ax, cmap):\n'
                       f'    x = np.array({x.tolist()})\n'
                       f'    y = np.array({y.tolist()})\n'
                       f'    u = np.array({u.tolist()})\n'
                       f'    v = np.array({v.tolist()})\n'
                       f'    return prt_flow_2d(x, y, u, v, fig, ax, {xlabel!r}, {ylabel!r}, {title!r}, cmap, {show}, False)\n')
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    s = np.sqrt(u**2+v**2)
    #lines = ax.streamplot(x,y,u,v,color=s,cmap=cmap)
    lines = ax.streamplot(x,y,u.T,v.T,color=s.T,cmap=cmap,**kwargs)
    ax.set_xlabel(xlabel = xlabel)
    ax.set_ylabel(ylabel = ylabel)
    ax.set_title(label = title)
    #ax.tick_params(labelsize = 'xx-large')
    cbar = fig.colorbar(lines.lines)
    #cbar.ax.tick_params(labelsize = 'xx-large')
    if show:
        plt.show()
    return None

def prt_dens_2d(
    x:np.ndarray, y:np.ndarray, 
    u:np.ndarray,
    fig:Figure|None = None, ax:Axes|None = None, 
    xlabel:str="", ylabel:str="", title:str="",
    cmap:str|Colormap=cmap_positive_ignorezero, show:bool = False, log:bool = False,
    **kwargs
    ):
    '''Draw contourf in given axes.
    
    ### Parameters:
    * `x`: `numpy.ndarray`. Same as the parameter `x` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `y`: `numpy.ndarray`. Same as the parameter `y` in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `u`: `numpy.ndarray`. 
    * `fig`: `matplotlib.figure.Figure|None`. If `fig` is `None`, this function will create a new figure and axes.
    * `ax`: `matplotlib.axes.Axes|None`. If `ax` is `None`, this function will create a new figure and axes.
    * `xlabel`, `ylabel`, `title`: `str`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `cmap`: `str|matplotlib.colors.Colormap`. Same as the parameter in `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.
    * `show`: `bool`. If the parameter is `True`, the function will call `matplotlib.pyplot.show()`. Default is `False`.
    * `log`: `bool`. If the parameter is `True`, the function will save the value of these parameters: `x`,`y`,`u`,`v`,`xlabel`,`ylabel`,`title`
    to a file in the 'log' folder.
    * `**kwargs`: other parameters that could pass to `matplotlib.pyplot.streamplot` or `matplotlib.pyplot.contourf`.'''
    if log:
        if os.path.exists('log'):
            if os.path.isfile('log'):
                raise OSError('Cannot create directory `log` for storing logs.')
        else:
            os.mkdir('log')
        timestamp = get_timestamp()
        with open(f'log\\{timestamp} Contourf.py', mode='x') as file:
            file.write('#!/usr/bin/python\n'
                       '# -*- coding: utf-8 -*-\n\n'
                       'from _plottools import prt_dens_2d\n'
                       'import numpy as np\n\n'
                       'def prt(fig, ax, cmap):\n'
                       f'    x = np.array({x.tolist()})\n'
                       f'    y = np.array({y.tolist()})\n'
                       f'    u = np.array({u.tolist()})\n'
                       f'    return prt_dens_2d(x, y, u, fig, ax, {xlabel!r}, {ylabel!r}, {title!r}, cmap, {show}, False)\n')
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    surf = ax.contourf(x, y, u.T, cmap=cmap, **kwargs)
    ax.set_xlabel(xlabel = xlabel)
    ax.set_ylabel(ylabel = ylabel)
    ax.set_title(label = title)
    #ax.tick_params(labelsize = 'xx-large')
    cbar = fig.colorbar(surf)
    #cbar.ax.tick_params(labelsize = 'xx-large')
    if show:
        plt.show()
    return None

def prt_mask_2d(
    ax:Axes,
    xRange:tuple[Real, Real],
    yRange:tuple[Real, Real],
    mask_func:Callable[[Real, Real], Real]):
    '''Draw mask in given axes.

    This function will draw a mask that covers `x, y` if `x, y` matches:
    * `mask_func(x,y) > 0`
    * `xRange[0] < x < xRange[1]`
    * `yRange[0] < y < yRange[1]`
    '''
    Nx = 1000
    Ny = 1000
    x = np.linspace(xRange[0], xRange[1], Nx)
    y = np.linspace(yRange[0], yRange[1], Ny)
    X, Y = np.meshgrid(x, y)
    try:
        Z:np.ndarray = mask_func(X, Y)
        if Z.shape != X.shape:
            raise Exception
    except:
        Z = np.vectorize(mask_func)(X, Y)
    
    ax.contourf(X, Y, Z, levels=[0, np.inf], colors=[(0.5,0.5,0.5,1)])
    ax.contour(X, Y, Z, levels=[0], colors="black")
    return None

# Rank Fig
class _ranksave(Save):
    def __init__(self, parent:"rankfig", prefix = "test_result", dir = "graphic"):
        self.parent = parent
        super().__init__(prefix, dir)

    def __call__(self, save_fig = True):
        for ax in self.parent.data:
            ax.add_refline()
        if self.parent.log is not None:
            with open(self.parent.log, mode='a') as file:
                file.write(f"\nfig.fig.tight_layout()\nfig.save.file_id = {self.file_id}\nfig.save()\n")
        return super().__call__(self.parent.fig, save_fig)

class rankfig:
    ''' A class which is used to draw figures that shows
    the error order of a method.
    
    ### Parameters
    * `rows`, `cols`, `sharex`, `sharey`, `supxlabel`, `supylabel`: Same as those parameters in `gridfig`
    * `title`: Same as the parameter `suptitle` in gridfig.
    * `prefix`, `dir`: Same as those parameters in `Save`.
    * `log`: `os.PathLike|str|bytes|None`. If `log` is not `None`, when operating on this instance, 
    it will store some key data at the location indicated by the parameter `log`.

    ### Usage
    ```
    r = rankfig(2, 1)   # a figure with two axes.

    r[0].add_line(
        h = np.array([0.1, 0.05, 0.025]),
        err = np.array([0.102, 0.0261, 0.0067]),
        label = 'test1'
    )   # add a line at the first axes.

    r[1].add_line(
        h = np.array([0.1, 0.05, 0.025]),
        err = np.array([0.105, 0.0502, 0.0231]),
        label = 'test1'
    )   # add a line at the second axes.

    r[0].refline_rank = (1, 2) # the first axes will draw two lines of reference, 
                               # one has the slope of 1 and one has the slope of 2.
                               
    r[1].refline_rank = None   # the second axes will draw no line of reference.

    r[0].set_xlabel('xlabel of the first axes') # set the xlabel of the first axes
    r[0].set_ylabel('ylabel of the first axes') # set the ylabel of the first axes
    r[1].set_title('title of the second axes')  # set the title of the second axes

    r.save()   # save the figure. 
               # If `prefix` and `dir` parameter remains the same when creating the instance.
               # the figure will be saved by default at `graphic\\test_result_rank_00000.png`
    r.show()   # show the figure.
    ```
    '''
    def __init__(self,
        rows:int = 1, 
        cols:int = 1, 
        sharex:str = "all", 
        sharey:str = "all",
        title:str = "",
        prefix:str = 'test_result_rank',
        dir:str = 'graphic', 
        supxlabel:str = "h",
        supylabel:str = "L2 Relative Error",
        log: os.PathLike|str|bytes|None = f"log\\{get_timestamp()} Rankfig.py"
        ):
        self.log = log
        if log is not None:
            with open(self.log, mode=('w' if os.path.exists(self.log) else 'x')) as file:
                file.write('#!/usr/bin/python\n'
                        '# -*- coding: utf-8 -*-\n\n'
                        'from _plottools import rankfig\n'
                        'import numpy as np\n\n'
                        'nan = float("nan")\n'
                        f'fig = rankfig({rows}, {cols}, {sharex!r}, {sharey!r}, {title!r}, {prefix!r}, {dir!r}, {supxlabel!r}, {supylabel!r}, None)\n')
        self.fig, axs = gridfig(rows, cols, sharex, sharey, title)
        if supxlabel != "":
            self.fig.supxlabel(supxlabel) #, x=0.9, horizontalalignment = "right")
        if supylabel != "":
            self.fig.supylabel(supylabel, x=0.0, horizontalalignment = "left", y=0.9, verticalalignment = "top")
        self.data = []
        for i in range(len(axs)):
            self.data.append(rankax(axs[i], log = self.log, index = i))
        self.save = _ranksave(parent = self, prefix = prefix, dir = dir)
        
    def show(self, show_fig:bool = True):
        '''Release the memory occupied by this object.
        
        ### Parameter
        * `show_fig`: `bool`. If the parameter is `True`, then it will show
        the figure before release the memory.'''
        if self.log is not None:
            with open(self.log, mode='a') as file:
                file.write(f'fig.show({show_fig})')
        if show_fig:
            for ax in self.data:
                ax.add_refline()
        show(self.fig, show_fig)
        del self
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> "rankax":
        return self.data[i]
  
class rankax:
    '''Type of the objects that can be getted by `rankfig.__getitem__()`.'''
    def __init__(self, ax:Axes, log:os.PathLike|str|bytes|None, index:int, 
                 refline_rank:tuple[int, ...]|int|None = 2):
        self.ax = ax
        self.log = log
        self.index = index
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.tick_params(axis='y')
        self.ax.tick_params(axis='x', which = 'both', rotation=30)
        # self.ax.xaxis.set_major_formatter('%f')
        # self.ax.set(xlabel=xlabel,ylabel=ylabel)

        self.line_color = -1
        self.refline_color = -1

        self.refline_rank = refline_rank
        self.h = None
        self.err_base_max = None
        self.err_base_min = None

    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
             'mediumblue', 'greenyellow', 'black', 'magenta', 'deeppink']

    marker = ['o', 'v', '^', '<', '>',
              '8', 's', 'p', '*', 'h',
              'H', 'D', 'd', 'P', 'X']

    refcolor = ['dimgray', 'rosybrown']

    def set_title(self, text:str = ""):
        '''Set the title to `text`.'''
        if self.log is not None:
            with open(self.log, mode='a') as file:
                file.write(f'fig[{self.index}].set_title({text!r})\n')
        self.ax.set_title(label=text)

    def set_xlabel(self, text:str = ""):
        '''Set the xlabel to `text`.'''
        if self.log is not None:
            with open(self.log, mode='a') as file:
                file.write(f'fig[{self.index}].set_xlabel({text!r})\n')
        self.ax.set_xlabel(xlabel=text)

    def set_ylabel(self, text:str = ""):
        '''Set the ylabel to `text`.'''
        if self.log is not None:
            with open(self.log, mode='a') as file:
                file.write(f'fig[{self.index}].set_ylabel({text!r})\n')
        self.ax.set_ylabel(ylabel=text)

    def add_line(self, h:np.ndarray, err:np.ndarray, label:str = '', line_style:str|tuple[int, tuple[int, ...]]|None = 'solid') -> float:
        '''Add lines about the step size `h` and the error `err` on the error graph 
        and return the error order.
        ### Parameters
        * `h`: `np.ndarray`. An array that stores the step size.
        * `err`: `np.ndarray`. An array that stores the error that related to the step size.
        * `label`: `str`. The discription that will show at the legend.
        * `line_style`: see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        '''
        if self.log is not None:
            with open(self.log, mode='a') as file:
                file.write(f'fig[{self.index}].add_line(\n    h=np.array({h.tolist()}),\n    err=np.array({err.tolist()}),\n'
                        f'    label={label!r},\n    line_style=')
                if line_style is None:
                    file.write('None')
                else:
                    file.write(str(line_style) if isinstance(line_style, tuple) else '"'+line_style+'"')
                file.write(")\n")
        
        if self.h is None:
            self.h = h
            self.err_base_min = err.min()
            self.err_base_max = err.max()
        else:
            if self.err_base_min > (newbase:=np.nanmin(err)): self.err_base_min = newbase
            if self.err_base_max < (newbase:=np.nanmax(err)): self.err_base_max = newbase
        
        if np.isnan(err).any():
            print("Caution: FOUND NAN.")

        try:
            lnh = np.log(h)
            lne = np.log(err)
            p2 = poly.fit(lnh, lne, deg=(0, 1))

            rank = p2(1) - p2(0)
        except:
            if self.log is not None:
                print(f'rankax.add_line failed, data has been stored to {self.log}')
            else:
                print(f'rankax.add_line failed. If you want to retry, here is the data:\n    {h=}\n    {err=}')
            return 0
        
        self.line_color += 1
        self.line_color %= len(self.color)
        
        if line_style is not None:
            self.ax.scatter(h, err, c = self.color[self.line_color], marker = self.marker[self.line_color], label=label)
            self.ax.plot(h, err, linestyle=line_style, color=self.color[self.line_color])
            #self.ax.plot(h, np.exp(p2(lnh)), label=label, linestyle=line_style, color=self.color[self.line_color])
        else:
            self.ax.scatter(h, err, c = self.color[self.line_color], marker = self.marker[self.line_color], label=label)
        self.ax.legend(loc='lower right')
        return rank
    
    def add_refline(self):
        if self.refline_rank is not None:
            if isinstance(self.refline_rank, int):
                self.refline_rank = (self.refline_rank,)
            for refline_rank in self.refline_rank:
                self._add_refline(self.h, refline_rank, label=f'slope = {refline_rank}', line_style = 'dashdot')
            self.refline_rank = None

    def _add_refline(self, h:np.ndarray, rank:float, label:str = '', line_style:str|tuple[int, tuple[int, ...]] = 'dashdot'):
        err = (h/self.h.min())**rank*self.err_base_min*(self.err_base_min/self.err_base_max)**0.05
        self.refline_color += 1
        self.refline_color %= len(self.refcolor)
        self.ax.plot(h, err, label=label, linestyle=line_style, color=self.refcolor[self.refline_color])
        self.ax.legend(loc='lower right')

def get_rank(
    h:np.ndarray, 
    err:np.ndarray, 
    ax:Axes = None,
    label:str = '') -> float:
    'Return the order.'
    lnh = np.log(h)
    lne = np.log(err)
    p2 = poly.fit(lnh, lne, deg=(0, 1))

    rank = p2(1) - p2(0)
    return rank

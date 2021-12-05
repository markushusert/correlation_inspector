import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import sys
import ipywidgets as widgets
from IPython.display import display
import glob
from functools import wraps
from PIL import Image
from ipywidgets import interact, interactive, fixed, interact_manual


g_in_jupyter='ipykernel' in sys.modules
def row_vectorize(f):
    @wraps(f)
    def wrapped_f(X,*args,**kwargs):
        X = np.asarray(X)
        rows = X.reshape(-1, X.shape[-1])
        for row in rows:
            return np.reshape([f(row,*args,**kwargs) for row in rows],X.shape[:-1] + (-1,))
    return wrapped_f
@row_vectorize
def distance(row,point):
    return math.sqrt((row[0]-point[0])**2+(row[1]-point[1])**2)

class correlation_inspector:
    def __init__(self,data,fields,inspectkey="control",image_path=None):
        self.data=data
        self.text=None
        self.scatter_points=None
        self.fields=fields
        self.image_path=image_path
        self.cor_coef=self.calc_correl()
        self.inspectkey=inspectkey
        self.key_pressed=False
        self.displayed_images=list()
        self.figure,self.matshow_ax,self.scatter_ax=self.create_interactive_correlation_fig()
        self.idxs_to_scatter=[None,None]
        if not g_in_jupyter:
            self.figure.show()
        self.display_field_selection_dropdown()
        #self.msg(f"looking for {self.inspectkey}")
    
    def on_hover(self,event):
        if event.inaxes is self.scatter_ax:
            scatterpoint_on_hover,coords_hovered_point=self.get_scatter_point_hovered((event.xdata,event.ydata))
            #text.set_text(f"hovering over{scatterpoint_on_hover}")
            #self.msg(f"hovering over point {scatterpoint_on_hover}, at pos {coords_hovered_point}")
            if scatterpoint_on_hover:
                #print(f"highlighting point:{scatterpoint_on_hover}")
                self.highlight_hovered_point(coords_hovered_point)
        
    def highlight_hovered_point(self,coords):
        
        if hasattr(self, "last_scatter"):
            #self.msg("removing scatter")
            self.last_scatter.remove()
        #text.set_text(f"scattering_coords:{[coords[0],coords[1]]}")
        self.last_scatter=self.scatter_ax.scatter(coords[0],coords[1],c=np.array([1,0,0]).reshape((1,3)))
    def get_scatter_point_hovered(self,cursor_pos):
        #cursor_pos in data coordinates
        #transform to display coords
        cursor_pos=self.scatter_ax.transData.transform(cursor_pos)
        if not len(self.scatter_ax.collections):
            return (None,None)
        cs=self.scatter_ax.collections[0]
        cs.set_offset_position('data')

        #point coordinates in display coordinates
        scatter_cords_data=cs.get_offsets()#Nx2 np-array, 1row for each point
        scatter_coords_display=self.scatter_ax.transData.transform(scatter_cords_data)
        
        scattersizes=self.scatter_points.get_sizes()
        closest_scatter_idx,distance=self.get_closest_point_to_cursor(scatter_coords_display,cursor_pos)
        size_of_closest_point=scattersizes if (scattersizes.size==1) else scattersizes[closest_scatter_idx]
        lim=math.sqrt(size_of_closest_point)
        #print(f"distance={distance},lim={lim}")
        if distance<lim:
            return closest_scatter_idx,scatter_cords_data[closest_scatter_idx,:]
        return (None,None)
    def get_closest_point_to_cursor(self,scatterpoints,cursor_pos):
        distance_points_to_cursor=distance(scatterpoints,cursor_pos)
        min_idx=np.argmin(distance_points_to_cursor)
        if False:
            print("scatterpoints")
            print(f"{scatterpoints}")
            print("cursor_pos")
            print(f"{cursor_pos}")
            print("distance")
            print(f"{distance_points_to_cursor}")
            print(f"minidx={min_idx}")
        return min_idx,distance_points_to_cursor[min_idx][0]
    def on_key_press(self,event):
        if event.key==self.inspectkey:
            self.key_pressed=True
        #self.msg(f"key pressed {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
        
    def on_key_release(self,event):
        if event.key==self.inspectkey:
            self.key_pressed=False
        #self.msg(f"key released {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
    def msg(self,txt):
        if g_in_jupyter:
            if self.text:
                self.text.set_text(txt)
            else:
                self.text=self.matshow_ax.text(1,1,txt)
        else:
            print(txt)
    def clicked_on_correl(self,event):
        self.idxs_to_scatter[0]=int(round(event.xdata))
        self.idxs_to_scatter[1]=int(round(event.ydata))
        self.plot_scatter()
    def clicked_on_scatter(self,event):
        clicked_idx,coords=self.get_scatter_point_hovered((event.xdata,event.ydata))
        #self.msg(f"clicked on scatter,idx={clicked_idx},coords={coords}")
        if clicked_idx:
            self.show_images(clicked_idx)
    def on_click(self,event):
        if not self.key_pressed:
            return
        msg=f"key_pressed:{self.key_pressed},time{datetime.datetime.now()}"
        #self.msg(msg)

        #if click on correlation matrix
        if event.inaxes is self.matshow_ax:
            self.clicked_on_correl(event)
            
        #or click on scatter-plot-axes
        elif event.inaxes is self.scatter_ax:
            self.clicked_on_scatter(event)
    def show_images(self,calc_id):
        #only show images if path is given in __init__
        if not self.image_path:
            return

        #close all exisiting images
        while len(self.displayed_images):
            self.displayed_images.pop().close()

        path_to_images=self.image_path.format(calc_id=calc_id)
        images_to_show=glob.glob(path_to_images)
        for image_path in images_to_show:
            self.displayed_images.append(Image.open(image_path))
            self.displayed_images[-1].show()
    
    def scatter_dropdown_interact_x(self,idx_val,idx_nr):
        self.idxs_to_scatter[idx_nr]=idx_val
        self.plot_scatter()

    def create_dropdown_widget(self,idx_nr):
        axis_to_set="x-axis" if idx_nr==0 else "y-axis"
        list_tuples_field_and_nr=[(field,nr) for nr,field in enumerate(self.fields)]
        return widgets.Dropdown(options=list_tuples_field_and_nr,description=f'{axis_to_set}:')


    def display_field_selection_dropdown(self):
        if not g_in_jupyter:
            return
        #for both x and y axis
        for idx_nr in range(2):
            dropdown_widget=self.create_dropdown_widget(idx_nr)
            interact(self.scatter_dropdown_interact_x,idx_val=dropdown_widget,idx_nr=fixed(idx_nr))
            

    
    def calc_correl(self):
        return np.corrcoef(self.data)
    def plot_scatter(self):
        
        self.scatter_ax.clear()
        if any(i is None for i in self.idxs_to_scatter):
            return
        #self.msg(f"xidx={xidx},yidx={self.idxs_to_scatter[1]}")
        name_x=self.fields[self.idxs_to_scatter[0]]
        name_y=self.fields[self.idxs_to_scatter[1]]
        xdata=self.data[self.idxs_to_scatter[0],:]
        ydata=self.data[self.idxs_to_scatter[1],:]
        self.scatter_points=self.scatter_ax.scatter(xdata,ydata)

        #self.msg(f"printing x,{xdat} vs y,{ydat}")
        self.scatter_ax.set_xlabel(f"{name_x}, row:{self.idxs_to_scatter[0]}")
        self.scatter_ax.set_ylabel(f"{name_y}, row:{self.idxs_to_scatter[1]}")
        self.set_lims(self.scatter_ax,xdata,ydata)
        #self.scatter_ax.draw()

    def set_lims(self,axes,xdata,ydata):
        
        minx=min(xdata)
        maxx=max(xdata)
        rangex=maxx-minx
        miny=min(ydata)
        maxy=max(ydata)
        rangey=maxy-miny
        fac=0.05
        #self.msg(f"setting lims:{[minx-fac*rangex,maxx+fac*rangex]}")
        axes.set_xlim(minx-fac*rangex,maxx+fac*rangex)
        axes.set_ylim(miny-fac*rangey,maxy+fac*rangey)

    def create_interactive_correlation_fig(self):
        #create figure
        fig=plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(ncols=2, nrows=1)

         #create axes 0 for plotting correlation-plot
        ax0=fig.add_subplot(spec[0])
        ax0.matshow(self.cor_coef)
        ax0.set_title(f"{self.inspectkey}+Click to inspect Scatter plot")
        ax1=fig.add_subplot(spec[1])
        #ax1.set_title(f"{self.inspectkey}+Click to inspect images of calculation")
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        return fig,ax0,ax1
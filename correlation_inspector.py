import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import sys
import ipywidgets as widgets
from IPython.display import display
import glob
import bokeh
from functools import wraps
from PIL import Image
from ipywidgets import interact, interactive, fixed, interact_manual
import panel as pn
pn.extension("tabulator")

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
    def __init__(self,data,fields,nr_inputs,inspectkey="control",image_path=None):
        if data.shape[0]!=len(fields):
            raise ValueError("fields and first dimension of data need to have same length")
        #setting of data attributes
        self.data=data
        self.cor_coef=self.calc_correl()
        self.text=None
        self.scatter_points=None
        self.nr_inputs=nr_inputs
        self.nr_outputs=data.shape[0]-nr_inputs
        self.fields=fields
        self.image_path=image_path
        self.inspectkey=inspectkey
        self.key_pressed=False
        self.displayed_images=list()
        self.figure,self.matshow_ax,self.scatter_ax=self.create_layout()
        self.idxs_to_scatter=[None,None]
        #need to set active_fields_list before plotting
        self.set_is_field_active_list([True for i in range(len(fields))])
        
        self.matshow_ax.matshow(self.get_cor_coef())
                
        if not g_in_jupyter:
            self.figure.show()
        self.display_field_selection_dropdown()
    
    def on_hover(self,event):
        if event.inaxes is self.scatter_ax:
            scatterpoint_idx_on_hover,coords_hovered_point=self.get_scatter_point_hovered((event.xdata,event.ydata))
            #text.set_text(f"hovering over{scatterpoint_on_hover}")
            #self.msg(f"hovering over point {scatterpoint_on_hover}, at pos {coords_hovered_point}")
            if scatterpoint_idx_on_hover:
                #print(f"highlighting point:{scatterpoint_on_hover}")
                self.highlight_hovered_point(coords_hovered_point,scatterpoint_idx_on_hover)
        
    def highlight_hovered_point(self,coords,idx):
        #coords in data coordinate 
        self.mark_point_red(coords)
        self.annotate_point(coords,idx)
    def annotate_point(self,coords,idx):
        if hasattr(self, "last_anot"):
            self.last_anot.remove()
        
        self.last_anot=self.scatter_ax.annotate(f"{idx}",coords)#bbox=dict(boxstyle="round",facecolor='wheat')

    def mark_point_red(self,coords):
        
        if hasattr(self, "last_scatter"):
            self.last_scatter.remove()
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
        xidx_clicked=int(round(event.xdata))
        yidx_clicked=int(round(event.ydata))
        self.idxs_to_scatter[0]=self.get_active_fields()[xidx_clicked]
        self.idxs_to_scatter[1]=self.get_active_fields()[yidx_clicked]
        self.dropdown_widgets[0].value=self.idxs_to_scatter[0]
        self.dropdown_widgets[1].value=self.idxs_to_scatter[1]
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
    
    def scatter_dropdown_interact(self,idx_val,idx_nr):
        self.idxs_to_scatter[idx_nr]=idx_val
        self.plot_scatter()
    def options_for_widget(self):
        active_field_labels=[self.fields[i] for i in self.get_active_fields()]
        return [(f"{self.get_active_fields()[nr]}:{field}",nr) for nr,field in enumerate(active_field_labels)]
    def create_dropdown_widget(self,idx_nr):
        axis_to_set="x-axis" if idx_nr==0 else "y-axis"
        list_tuples_field_and_nr=self.options_for_widget()
        return widgets.Dropdown(options=list_tuples_field_and_nr,description=f'{axis_to_set}:')
    def create_swap_button(self):
        tooltip="swaps x- and y-Axis of Scaterplot"
        button=widgets.Button(description='Swap',button_style='',tooltip=tooltip)
        button.on_click(self.swap_fun)
        return button
    def create_filter_widget(self):

        widget=widgets.FloatText(value=0.0,description="filter by correlation")
        to_display=interactive(self.filter_fields,lim_abs_correl=widget)
        return to_display
    def swap_fun(self,button):
        #self.idxs_to_scatter.reverse()
        temp=self.dropdown_widgets[0].value
        self.dropdown_widgets[0].value=self.dropdown_widgets[1].value
        self.dropdown_widgets[1].value=temp
        self.plot_scatter()   
    def filter_fields(self,lim_abs_correl):
        self.correl_overview_dataframe["is_active"]=self.correl_overview_dataframe["max_abs_correl"]>=lim_abs_correl
        self.update_active_fields()
    def display_field_selection_dropdown(self):
        if not g_in_jupyter:
            return
        #for both x and y axis
        self.dropdown_widgets=[]
        for idx_nr in range(2):
            self.dropdown_widgets.append(self.create_dropdown_widget(idx_nr))
            to_display=interactive(self.scatter_dropdown_interact,idx_val=self.dropdown_widgets[-1],idx_nr=fixed(idx_nr))
            display(to_display)
        self.swap_button=self.create_swap_button()
        display(self.swap_button)
    def calc_correl(self):
        return np.corrcoef(self.data)
    def plot_scatter(self):
        for artist in self.scatter_ax.collections:
            artist.remove()
        if any(i is None for i in self.idxs_to_scatter):
            return
        #self.msg(f"xidx={xidx},yidx={self.idxs_to_scatter[1]}")
        name_x=self.fields[self.idxs_to_scatter[0]]
        name_y=self.fields[self.idxs_to_scatter[1]]
        xdata=self.data[self.idxs_to_scatter[0],:]
        ydata=self.data[self.idxs_to_scatter[1],:]
        self.scatter_points=self.scatter_ax.scatter(xdata,ydata,c="blue")

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
    def set_is_field_active_list(self,new_list):
        #new_list=iterable of bool, size of self.fields
        self.is_field_active=list(new_list)
        self.active_fields=[i for i in range(len(self.fields)) if new_list[i]]
        self.inactive_fields=[i for i in range(len(self.fields)) if not new_list[i]]
        if hasattr(self,"correl_overview_dataframe"):
            self.correl_overview_dataframe["is_active"]=self.is_field_active
        if hasattr(self,"dropdown_widgets"):
            for widget in self.dropdown_widgets:
                widget.options=self.options_for_widget()
    def get_nr_inputs(self):
        return self.nr_inputs
    def get_cor_coef(self):
        return self.cor_coef[self.get_active_fields(),:][:,self.get_active_fields()]
    def get_active_inputs(self):
        return [i for i in self.active_fields if i < self.nr_inputs]
    def get_active_fields(self):
        return self.active_fields
    def get_inactive_fields(self):
        return self.inactive_fields
    def get_active_ouputs(self):
        return [i for i in self.active_fields if i >= self.nr_inputs]
    def allocate_empty_overview_df(self):
        return pd.DataFrame(
            {
                "name":self.fields,
                #not needed since tabulator provides it
                #"idx":[i for i in range(len(self.fields))],
                "max_correl":0.0,
                "max_idx":0,
                "max_name":"",
                "min_correl":0.0,
                "min_idx":0,
                "min_name":"",
                "max_abs_correl":0.0,
                "is_active":True
            }
        )
    def calc_correl_overview(self):
        if not hasattr(self,"correl_overview_dataframe"):
            raise Exception(f"dataframe does not exist as attribute of {self}")
        active_outputs=self.get_active_ouputs()
        active_inputs=self.get_active_inputs()
        #correl_input_to_output: rows contain ALL available inputs, cols contain ONLY active outputs
        correl_input_to_output=self.cor_coef[:self.nr_inputs,active_outputs]
        #correl_input_to_output: rows contain ALL available outputs, cols contain ONLY active inputs
        correl_output_to_input=self.cor_coef[self.nr_inputs:,active_inputs]
        for i in range(2):
            if i==0:
                #evaluate input data
                data_frame_to_set=self.correl_overview_dataframe[:][:self.nr_inputs]
                correl_data_to_eval=correl_input_to_output
                idx_to_correlate_against=active_outputs
            elif i==1:
                #evaluate output data
                data_frame_to_set=self.correl_overview_dataframe[:][self.nr_inputs:]
                correl_data_to_eval=correl_output_to_input
                idx_to_correlate_against=active_inputs
            #later
            data_frame_to_set["max_correl"]=np.amax(correl_data_to_eval,1)
            data_frame_to_set["max_idx"]=[idx_to_correlate_against[i] for i in np.argmax(correl_data_to_eval,1)]
            data_frame_to_set["max_name"]=[self.fields[i] for i in data_frame_to_set["max_idx"]]
            data_frame_to_set["min_correl"]=np.amin(correl_data_to_eval,1)
            data_frame_to_set["min_idx"]=[idx_to_correlate_against[i] for i in np.argmin(correl_data_to_eval,1)]
            data_frame_to_set["min_name"]=[self.fields[i] for i in data_frame_to_set["min_idx"]]
            data_frame_to_set["max_abs_correl"]=np.maximum(data_frame_to_set["max_correl"],-data_frame_to_set["min_correl"])
    def create_tabulator(self,active=True):
        #make only the is_active tab editable for the user
        immutable_fields=[field for field in self.correl_overview_dataframe.columns.values.tolist() if field not in {"is_active"}]
        editors_to_use={name:bokeh.models.widgets.tables.CellEditor() for name in immutable_fields}
        formatters_to_use={"is_active":{"type":"tickCross"}}        
        return pn.widgets.Tabulator(self.correl_overview_dataframe,frozen_columns=[0],
            editors=editors_to_use,formatters=formatters_to_use)
    def update_active_fields(self,*args):#use *args to allow passing button as arg which is required by widget
        #read active fields from dataframe
        active_fields=self.correl_overview_dataframe["is_active"]
        self.set_is_field_active_list(active_fields)
        self.matshow_ax.matshow(self.get_cor_coef())
        #self.show_tabulator()

    def create_update_button(self):
        tooltip="updates correlation_plot to only show active fields"
        button=widgets.Button(description='Update',button_style='',tooltip=tooltip)
        button.on_click(self.update_active_fields)
        return button
    def show_spreadsheet_view(self):
        if not hasattr(self,"correl_overview_dataframe"):
            self.correl_overview_dataframe=self.allocate_empty_overview_df()
        self.calc_correl_overview()
        self.filter_widget=self.create_filter_widget()
        display(self.filter_widget)
        self.update_button=self.create_update_button()
        display(self.update_button)

        self.show_tabulator()
    def show_tabulator(self):
        self.tabulator=self.create_tabulator()
        #color rows green for inputs and red for outputs
        def fmt_fun(x):
            return ['background: lightgreen' if x.name<self.nr_inputs else 'background: #FFA07A' for i in x]
        self.tabulator.style.apply(fmt_fun, axis=1)
        display(self.tabulator)
    def create_layout(self):
        #create figure
        fig=plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(ncols=2, nrows=1)

         #create axes 0 for plotting correlation-plot
        ax0=fig.add_subplot(spec[0])
        ax0.set_title(f"{self.inspectkey}+Click to inspect")
        ax1=fig.add_subplot(spec[1])
        #ax1.set_title(f"{self.inspectkey}+Click to inspect images of calculation")
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        return fig,ax0,ax1

    
        
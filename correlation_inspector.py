import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from collections import OrderedDict
from pyexcel_ods import save_data
from pyexcel_ods import get_data
out=widgets.Output()
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
    def __init__(self,data,fields,nr_inputs,inspectkey="control",image_path=None,ods_path=None):
        if data.shape[0]!=len(fields):
            raise ValueError("fields and first dimension of data need to have same length")
        #setting of data attributes
        self.data=data
        self.cor_coef=self.calc_correl()
        self.text=None
        self.scatter_points=None
        self.nr_inputs=nr_inputs
        self.nr_outputs=data.shape[0]-nr_inputs
        self.fieldnames=fields
        self.image_path=image_path
        self.inspectkey=inspectkey
        self.key_pressed=False
        self.displayed_images=list()
        self.figure,self.matshow_ax,self.scatter_ax=self.create_layout()
        self.idxs_to_scatter=[None,None]
        self.ods_path=ods_path
        self.marked_correls=np.zeros(self.cor_coef.shape, dtype=bool)#array of only False
        if ods_path:
            self.read_ods()
        else:
            self.ods_data=OrderedDict()
            
        #need to set active_fields_list before plotting
        self.set_is_field_active_list([True for i in range(len(fields))])
        
        self.matshow_ax.matshow(self.get_cor_coef())
                
        if not g_in_jupyter:
            self.figure.show()
        self.display_field_selection_dropdown()
    def read_ods(self):
        self.ods_data=get_data(self.ods_path)
        for row in self.ods_data.get("Sheet1",list()):
            if len(row)>=2:
                if isinstance(row[0], int) and isinstance(row[2], int):        
                    self.marked_correls[row[0],row[2]]=True
    def write_ods(self):
        with out:
            print(f"writing data:{self.ods_data} to {self.ods_path}")
        save_data(self.ods_path, self.ods_data)

    def on_hover(self,event):
        """
        handles the event of the mouse hovering over the scatterplot

        highlights the point being hovered over
        """
        if event.inaxes is self.scatter_ax:
            scatterpoint_idx_on_hover,coords_hovered_point=self.get_scatter_point_hovered((event.xdata,event.ydata))
            #text.set_text(f"hovering over{scatterpoint_on_hover}")
            #self.msg(f"hovering over point {scatterpoint_on_hover}, at pos {coords_hovered_point}")
            if scatterpoint_idx_on_hover is not None:
                #print(f"highlighting point:{scatterpoint_on_hover}")
                self.highlight_hovered_point(coords_hovered_point,scatterpoint_idx_on_hover)
        
    def highlight_hovered_point(self,coords,idx):
        """
        coords: iterable of length 2, coordinates of point in data-coordinate-system
        idx: idx of point, indicating the number of the calculation it belongs to
        """
        #coords in data coordinate
        print(f"highlighting point with coords {coords} at idx {idx}")
        self.mark_point_red(coords)
        self.annotate_point(coords,idx)
    def annotate_point(self,coords,idx):
        """
        annotate the point at coords by showing its idx next to it

        coords: iterable of length 2, coordinates of point in data-coordinate-system
        idx: idx of point, indicating the number of the calculation it belongs to
        """
        if hasattr(self, "last_anot"):
            if self.last_anot.get_figure():
                self.last_anot.remove()
        
        self.last_anot=self.scatter_ax.annotate(f"{idx}",coords)#bbox=dict(boxstyle="round",facecolor='wheat')

    def mark_point_red(self,coords):
        """
        highlight the scatterpoint by ploting a red point at its exact position
        """
        if hasattr(self, "last_scatter"):
            if self.last_scatter.get_figure():#only if last scatter has not been deleted after redrawing plot
                self.last_scatter.remove()
        self.last_scatter=self.scatter_ax.scatter(coords[0],coords[1],c=np.array([1,0,0]).reshape((1,3)))

    def get_scatter_point_hovered(self,cursor_pos):
        """
        returns index and coordinates of hovered point, or (None,None)

        cursor_pos: iterable of length 2 of the cursor in data-coordinates

        returns index(integer) and point_coords (iterable of length 2 in data-coordinates)
        returns (None,None) if no point is hovered
        """

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
        print(f"point {closest_scatter_idx},coords {scatter_cords_data[closest_scatter_idx,:]},distance={distance},lim={lim}")
        if distance<lim:
            return closest_scatter_idx,scatter_cords_data[closest_scatter_idx,:]
        return (None,None)
    def get_closest_point_to_cursor(self,scatterpoints,cursor_pos):
        """
        returns idx and distance scatterpoint closest to the cursor
        
        scatterpoints=Nx2 array of point coordinates in display-coordinate-system
        cursor_pos: iterable of length 2 of coordinates in display-coordinate-system
        
        returns idx as int and distance in display-coordinate-system
        """
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
        """
        keeps track of wether the designated key is held down or not
        """
        if event.key==self.inspectkey:
            self.key_pressed=True
        #self.msg(f"key pressed {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
        
    def on_key_release(self,event):
        """
        keeps track of wether the designated key is held down or not
        """
        if event.key==self.inspectkey:
            self.key_pressed=False
        #self.msg(f"key released {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
    def msg(self,txt):
        """
        display debugging message

        if code is run in jupyter-notebook, stdout of events may not be available
        then display text in a textbox
        """
        if g_in_jupyter:
            if self.text:
                self.text.set_text(txt)
            else:
                self.text=self.matshow_ax.text(1,1,txt)
        else:
            print(txt)
    def clicked_on_correl(self,event):
        """
        handles the event of clicking on the correlation plot

        detects which fields were clicked and shows their scatter-plot
        """
        xidx_clicked=int(round(event.xdata))
        yidx_clicked=int(round(event.ydata))
        self.idxs_to_scatter[0]=self.get_active_fields()[xidx_clicked]
        self.idxs_to_scatter[1]=self.get_active_fields()[yidx_clicked]
        self.dropdown_widgets[0].value=self.idxs_to_scatter[0]
        self.dropdown_widgets[1].value=self.idxs_to_scatter[1]
        self.plot_scatter()
    def clicked_on_scatter(self,event):
        """
        handles the event of clicking on the scatter-plot

        if a point has been clicked display the corresponding images
        """
        clicked_idx,coords=self.get_scatter_point_hovered((event.xdata,event.ydata))
        #self.msg(f"clicked on scatter,idx={clicked_idx},coords={coords}")
        if clicked_idx:
            self.show_images(clicked_idx)
    def on_click(self,event):
        """
        handles any click on the plot-figure

        delegates event to correlation-plot or scatter-plot if required
        """
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
        """
        displays the images for a given calculation-id (int)
        """
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
        """
        updates the fields currently shown in scatter plot, to be controlled via dropdown-widget

        idx_val=index of the field to be shown
        idx_nr=0 or 1 to determine which Axis is interacted with,0->x-Axis; 1->y-Axis
        """
        self.idxs_to_scatter[idx_nr]=idx_val
        self.plot_scatter()
    def options_for_widget(self):
        """
        creates the option list for the dropdown-widget depending on currently active fields
        """
        active_field_labels=[self.fieldnames[i] for i in self.get_active_fields()]
        return [(f"{self.get_active_fields()[nr]}:{field}",self.get_active_fields()[nr]) for nr,field in enumerate(active_field_labels)]
    def create_dropdown_widget(self,idx_nr):
        """
        creates dropdown-widget to control the displayed fields in the scatterplot

        idx_nr idx_nr=0 or 1 to determine which Axis is interacted with,0->x-Axis; 1->y-Axis
        """
        axis_to_set="x-axis" if idx_nr==0 else "y-axis"
        list_tuples_field_and_nr=self.options_for_widget()
        return widgets.Dropdown(options=list_tuples_field_and_nr,description=f'{axis_to_set}:')
    def update_mark_checkbox_button(self):
        """
        updates checkbox when new fields are chosen to be plotted
        """
        is_marked=self.marked_correls[self.idxs_to_scatter[0],self.idxs_to_scatter[1]]
        #temporary unobserve so that self.mark_checkbox_clicked not executed
        #and nothing written to ods
        self.checkbox.unobserve(self.mark_checkbox_clicked,names="value")
        self.checkbox.value=bool(is_marked)
        self.checkbox.observe(self.mark_checkbox_clicked,names="value")
    def group_similar_outputs(self,lim=0.8,reduce_fun=max):
        """
        returns list of lists of indices of fields correlated to on another
        """
        groups=[[self.nr_inputs]]#already fill first group with first output
        #thats why start loop at self.nr_inputs+1
        for i in range(self.nr_inputs+1,self.nr_inputs+self.nr_outputs):
            correls=[]
            for group in groups:
                correls.append(reduce_fun([abs(self.cor_coef[i,j])>lim for j in group]))
            maxidx=np.argmax(np.array(correls))
            if correls[maxidx]>lim:
                groups[maxidx].append(i)
            else:
                groups.append([i])
        return groups
    def get_highest_input_correl_of_groups(self,groups):
        correls=[]
        for group in groups:
            highest_correl_to_input=np.amax(self.cor_coef[:self.nr_inputs,group])
            correls.append(highest_correl_to_input)
        return correls
    def create_mark_checkbox_button(self):
        """
        creates a checkbox which marks import scatter-plots for later evaluation, if not already existent
        """
        is_marked=False
        
        tooltip="marks import scatter-plots for later evaluation"
        checkbox=widgets.Checkbox(description='Mark',tooltip=tooltip,value=is_marked)
        checkbox.observe(self.mark_checkbox_clicked,names='value')
        return checkbox
    def mark_checkbox_clicked(self,checkboxdict):
        self.set_marked_value([self.idxs_to_scatter[0],self.idxs_to_scatter[1]],checkboxdict['owner'].value)
        self.write_ods()
    def set_marked_value(self,list_idx,value):
        self.marked_correls[list_idx[0],list_idx[1]]=value
        def sort_key(row):
            if len(row)<1 or isinstance(row[0],str) or isinstance(row[2],str):
                return (0,0)
            else:
                return(row[0],row[2])
        if value==True:#add to marked data
            new_row=[list_idx[0],self.fieldnames[list_idx[0]],list_idx[1],self.fieldnames[list_idx[1]],float(self.cor_coef[list_idx[0],list_idx[1]])]
            self.ods_data["Sheet1"].append(new_row)
            self.ods_data["Sheet1"].sort(key=sort_key)
        else:#remove from it
            self.ods_data["Sheet1"]=[row for row in self.ods_data["Sheet1"] if not (len(row)>=2 and row[0]==list_idx[0] and row[2]==list_idx[1])]
        
        


    def create_swap_button(self):
        """
        creates a button which swaps the x and y Axis of the scatterplot
        """
        tooltip="swaps x- and y-Axis of Scaterplot"
        button=widgets.Button(description='Swap',button_style='',tooltip=tooltip)
        button.on_click(self.swap_fun)
        return button
    def create_filter_widget(self):
        """
        creates widget to interact with the filter-function
        """
        widget=widgets.FloatText(value=0.0,description="filter by correlation:")
        to_display=interactive(self.filter_fields,lim_abs_correl=widget)
        return to_display
    def swap_fun(self,button):
        """
        swaps values in dropdown-widgets axes of scatter-plot
        button argument is required by widget-module but unused
        """
        #self.idxs_to_scatter.reverse()
        temp=self.dropdown_widgets[0].value
        self.dropdown_widgets[0].value=self.dropdown_widgets[1].value
        self.dropdown_widgets[1].value=temp
        self.plot_scatter()   
    def filter_fields(self,lim_abs_correl):
        """
        filters irrelevant fields by setting them inactive
        if their maximum absolute correlation is below a given limit
        """
        self.correl_overview_dataframe["is_active"]=self.correl_overview_dataframe["max_abs_correl"]>=lim_abs_correl
        self.update_active_fields()
    def display_field_selection_dropdown(self):
        """
        creates and displays 2 dropdown widgets and a swap button to control the scatterplot
        """
        if not g_in_jupyter:
            return
        #for both x and y axis
        self.dropdown_widgets=[]
        to_display=[]
        for idx_nr in range(2):
            self.dropdown_widgets.append(self.create_dropdown_widget(idx_nr))
            to_display.append(interactive(self.scatter_dropdown_interact,idx_val=self.dropdown_widgets[-1],idx_nr=fixed(idx_nr)))
            #display(to_display)
        self.swap_button=self.create_swap_button()
        self.checkbox=self.create_mark_checkbox_button()
        display(widgets.HBox(to_display))
        display(widgets.HBox([self.swap_button,self.checkbox]))
    def calc_correl(self):
        """
        calculates correlation coefficients of provided data
        
        data-rows stand for a field
        data-cols stand for a specimen
        """
        return np.corrcoef(self.data)
    def plot_scatter(self,ret=False):
        """
        draws a scatter plot of 2 currently selected fields against each other
        """
        #remove any old scatterplots
        while len(self.scatter_ax.collections):
            self.scatter_ax.collections[0].remove()
        #remove any old annotations
        anotations=[i for i in self.scatter_ax.get_children() if isinstance(i,mpl.text.Annotation)]
        for annot in anotations:
            annot.remove()
        if ret: return
        #only proceed if both fields to scatter are given
        if any(i is None for i in self.idxs_to_scatter):
            return
        #self.msg(f"xidx={xidx},yidx={self.idxs_to_scatter[1]}")
        name_x=self.fieldnames[self.idxs_to_scatter[0]]
        name_y=self.fieldnames[self.idxs_to_scatter[1]]
        xdata=self.data[self.idxs_to_scatter[0],:]
        ydata=self.data[self.idxs_to_scatter[1],:]
        self.scatter_points=self.scatter_ax.scatter(xdata,ydata,c="blue")

        #self.msg(f"printing x,{xdat} vs y,{ydat}")
        self.scatter_ax.set_xlabel(f"{name_x}, row:{self.idxs_to_scatter[0]}")
        self.scatter_ax.set_ylabel(f"{name_y}, row:{self.idxs_to_scatter[1]}")
        self.set_lims(self.scatter_ax,xdata,ydata)
        self.scatter_ax.set_title(f"correlation:{self.cor_coef[self.idxs_to_scatter[0],self.idxs_to_scatter[1]]}")
        #self.scatter_ax.draw()
        self.update_mark_checkbox_button()
    def set_lims(self,axes,xdata,ydata):
        """
        set the limits for the given axes, so that all datapoints can be seen

        xdata/ydata given in data-coordinate-system
        """
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
        """
        updates active status and all its dependencies
        ONLY use this function to change the active status of a field!

        new_list=iterable of bools, one for each field. True->active, False->inactive
        """
        #new_list=iterable of bool, size of self.fieldnames
        self.is_field_active=list(new_list)
        self.active_fields=[i for i in range(len(self.fieldnames)) if new_list[i]]
        self.inactive_fields=[i for i in range(len(self.fieldnames)) if not new_list[i]]
        if hasattr(self,"correl_overview_dataframe"):
            self.correl_overview_dataframe["is_active"]=self.is_field_active
            self.calc_correl_overview()
        if hasattr(self,"dropdown_widgets"):
            for widget in self.dropdown_widgets:
                widget.options=self.options_for_widget()
        if hasattr(self,"tabulator"):
            self.update_tabulator()
            self.style_tabulator()
    def update_tabulator(self):
        """
        updates tabulator-widget by patching each column of the underlying pd.DataFrame
        """
        for field in self.correl_overview_dataframe.columns:
            self.update_tabulator_field(field,self.correl_overview_dataframe[field])
    def update_tabulator_field(self,field,values):
        """
        updates a column of the tabulator-widget via patching

        field=name of column(str)
        values=iterable of values to set
        """
        patch_dict={field:[(i,val) for i,val in enumerate(values)]}
        #print(f"updating:{patch_dict}")
        self.tabulator.patch(patch_dict)
    def get_nr_inputs(self):
        return self.nr_inputs
    def get_cor_coef(self):
        """
        returns only active rows/columns of correlation-matrix
        """
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
        """
        create Dataframe for overview with default-data
        """
        return pd.DataFrame(
            {
                "name":self.fieldnames,
                #not needed since tabulator provides it
                #"idx":[i for i in range(len(self.fieldnames))],
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
        """
        fills Dataframe with statistical data of the correlation-matrix

        only consider inputs vs active outputs
        and outputs vs active inputs
        """
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
            
            data_frame_to_set["max_correl"]=np.amax(correl_data_to_eval,1)
            data_frame_to_set["max_idx"]=[idx_to_correlate_against[i] for i in np.argmax(correl_data_to_eval,1)]
            data_frame_to_set["max_name"]=[self.fieldnames[i] for i in data_frame_to_set["max_idx"]]
            data_frame_to_set["min_correl"]=np.amin(correl_data_to_eval,1)
            data_frame_to_set["min_idx"]=[idx_to_correlate_against[i] for i in np.argmin(correl_data_to_eval,1)]
            data_frame_to_set["min_name"]=[self.fieldnames[i] for i in data_frame_to_set["min_idx"]]
            data_frame_to_set["max_abs_correl"]=np.maximum(data_frame_to_set["max_correl"],-data_frame_to_set["min_correl"])
    
    def create_tabulator(self,active=True):
        """
        creates tabulator to display the Dataframe
        all columns except the is_active one are made ineditable
        """
        #make only the is_active tab editable for the user
        immutable_fields=[field for field in self.correl_overview_dataframe.columns.values.tolist() if field not in {"is_active"}]
        editors_to_use={name:bokeh.models.widgets.tables.CellEditor() for name in immutable_fields}
        formatters_to_use={"is_active":{"type":"tickCross"}}        
        return pn.widgets.Tabulator(self.correl_overview_dataframe,frozen_columns=[0],
            editors=editors_to_use,formatters=formatters_to_use)
    def update_active_fields(self,*args):#use *args to allow passing button as arg which is required by widget
        """
        reads active fields out of DataFrame and updates displays accordingly
        """
        #read active fields from dataframe
        active_fields=self.correl_overview_dataframe["is_active"]
        self.set_is_field_active_list(active_fields)
        self.matshow_ax.matshow(self.get_cor_coef())
        #self.show_tabulator()

    def create_update_button(self):
        """
        return a button to interact with the update_active_fields-function
        """
        tooltip="updates correlation_plot to only show active fields"
        button=widgets.Button(description='Update',button_style='',tooltip=tooltip)
        button.on_click(self.update_active_fields)
        return button
    def print_highest_correlation(self,idx,n=5):
        """
        prints the n highest correlation partners of the field specified by idx
        """
        if idx<self.nr_inputs:
            to_eval=np.abs(self.cor_coef)[idx,self.nr_inputs:]#evaluate given input vs all outputs
            offset=self.nr_inputs
        else:
            to_eval=np.abs(self.cor_coef)[idx,:self.nr_inputs]#evaluate given output vs all inputs
            offset=0
        #print(to_eval)
        highest_idx=np.argpartition(to_eval,-n)[-n:]
        #print(highest_idx)
        highest_idx=highest_idx[np.argsort(to_eval[highest_idx])]+offset#sort indizes low to high
        for i in highest_idx:
            print(f"correlation to field {i}:{self.fieldnames[i]} is {self.cor_coef[idx,i]}")

    def show_spreadsheet_view(self):
        """
        calculates the correlation-overview-DataFrame and displays it alongside
        a button for updateing and a field to interact with the filter-function
        """
        if not hasattr(self,"correl_overview_dataframe"):
            self.correl_overview_dataframe=self.allocate_empty_overview_df()
        self.calc_correl_overview()
        self.filter_widget=self.create_filter_widget()
        display(self.filter_widget)
        self.update_button=self.create_update_button()
        display(self.update_button)

        self.show_tabulator()
    
    def show_tabulator(self):
        """
        displays a pn.widgets.tabulator to interact with overview-DataFrame
        """
        self.tabulator=self.create_tabulator()
        self.style_tabulator()
        display(self.tabulator)
    def style_tabulator(self):
        return#no more styling since it gets removed by patching
        #color rows green for inputs and red for outputs
        def fmt_fun(x):
            print(f"x_name{x.name}")
            return ['background: lightgreen' if x.name<self.nr_inputs else 'background: #FFA07A' for i in x]
        self.tabulator.style.apply(fmt_fun, axis=1)
        
    def create_layout(self):
        """
        creates layout for the correlation- and scatter-plot
        """
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

    
        

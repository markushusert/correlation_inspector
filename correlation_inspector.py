import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
class correlation_inspector:
    def on_key_press(self,event):
        if event.key==self.inspectkey:
            self.key_pressed=True
        #self.msg(f"key pressed {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
        
    def on_key_release(self,event):
        if event.key==self.inspectkey:
            self.key_pressed=False
        #self.msg(f"key released {event.key}, looking for {self.inspectkey}, {self.key_pressed}")
    def msg(self,txt):
        if self.text:
            self.text.set_text(txt)
        else:
            self.text=self.matshow_ax.text(1,1,txt)
    def onclick(self,event):
        
        msg=f"key_pressed:{self.key_pressed},time{datetime.datetime.now()}"
        #self.msg(msg)
        if self.key_pressed:
            xdat=math.floor(event.xdata)
            ydat=math.floor(event.ydata)
            self.plot_scatter(xdat,ydat)
    def __init__(self,data,fields,inspectkey="control"):
        self.data=data
        self.text=None
        self.fields=fields
        self.cor_coef=self.calc_correl()
        self.inspectkey=inspectkey
        self.key_pressed=False
        self.figure,self.matshow_ax,self.scatter_ax=self.create_interactive_correlation_fig()
        #self.figure.show() only show when not in jupyter
        #self.msg(f"looking for {self.inspectkey}")
    def calc_correl(self):
        return np.corrcoef(self.data)
    def plot_scatter(self,xdat,ydat):
        self.scatter_ax.clear()
        name_x=self.fields[xdat]
        name_y=self.fields[ydat]
        self.scatter_ax.scatter(self.data[xdat,:],self.data[ydat,:])
        self.scatter_ax.set_xlabel(name_x)
        self.scatter_ax.set_ylabel(name_y)
    
    def create_interactive_correlation_fig(self):
        #create figure
        fig=plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(ncols=2, nrows=1)

         #create axes 0 for plotting correlation-plot
        ax0=fig.add_subplot(spec[0])
        ax0.matshow(self.cor_coef)
        ax1=fig.add_subplot(spec[1])
        self.a1=fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.a2=fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.a3=fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        return fig,ax0,ax1
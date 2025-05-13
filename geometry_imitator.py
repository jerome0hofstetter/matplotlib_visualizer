import numpy as np
import matplotlib.pyplot as plt
from shapes_mockup import *
from matplotlib.widgets import Slider,Button,CheckButtons

#### --------------------------------
"""
defines a geometry roughly as needing to extend the GeometryImitator class, returning  list of shapes 
and what kind of adjustable prameters are needed, then this geometry imitator can be started to open a matplotlib plot with another window for sliders
"""
#### --------------------------------

def paint_shapes_in_axis(shapes : list[Shape],ax,xlim,ylim,num=100 ):
    x = np.linspace(*xlim, num)
    y = np.linspace(*ylim, num)

    X, Y = np.meshgrid(x, y)
    ax.set_aspect('equal') 
    ax.set_autoscale_on(False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for shape in shapes:
        if shape:
            shape.paint_shape(X,Y,ax)
            shape.after_drawing_shape(ax)
    legend_patches = [
        shape.get_label_patch() for shape in shapes if shape
    ]
    h, l = ax.get_legend_handles_labels()
    # Add legend
    ax.legend(handles=legend_patches+h, title="Volumes")

EMPTY_PARAM = ("",0,0,0,"")
REDRAW_PARAM = ("redraw",0,0,0,"Redraw")
RASTER_NUM_PARAM = ("raster",150,30,1000,"Raster Number")
class GeometryImitator(ABC):
    """
    extend this class, implement the abstract methods

    get an instance of it, call create_sliders, and append_redraw
    or directly use view_geometry
    """

    def __init__(self, view_size,see_negative_x=False,see_negative_y=False):
        self.view_size = view_size
        self.see_negative_y = see_negative_y
        self.see_negative_x = see_negative_x
        self.button = None

    def after_init(self):
        pass
    
    def get_geometry_values(self):
        """
        return nothing or a list of tuples of the form (name,val,postfix) all of them being strings
        """
        pass

    def get_geometry_values_string(self):
        if not self.get_geometry_values():
            return ""
        
        result = ""
        for (name,val,postfix) in self.get_geometry_values():
            result += f"\n{name:<20}: {val:>10} {postfix:<6}"
        return result

    def get_title(self):
        return "Geometry" + self.get_geometry_values_string()

    def raster_count(self):
        return int(self[RASTER_NUM_PARAM[0]]) if RASTER_NUM_PARAM[0] in self.parameter_dict else 100

    def get_init_xlim(self):
        return (-self.view_size if self.see_negative_x else 0, self.view_size )
    def get_init_ylim(self):
        return (-self.view_size if self.see_negative_y else 0, self.view_size )
        
    @abstractmethod
    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        """
        returns a list of entries of the form (parameter name, initval, minval,maxval,label)
        
        initialvalue being a float gives a slider, it being a boolean a checkbox
        """
        return [("a",1,1,2,"label")]
    
    @abstractmethod
    def get_shape_list(self) -> list[Shape]:
        pass
    

    def get_slider_dict(self):
        return self.parameter_dict

    def __getitem__(self, key):
        obj = self.parameter_dict[key]
        if isinstance(obj,CheckButtons):
            return obj.get_status()[0]
        return obj.val
    
    def create_sliders(self, fig , startindex=0,index_max=-1,cols = 1):
        """
        given startindex and max index and plt figure, create slides for the parameters given from get_parameter_list
        """
        self.parameter_dict= dict()
        param_list = self.get_parameter_list()
        index_max = len(param_list) if index_max==-1 else index_max
        rows = (index_max-startindex-1)//cols + 1
        gs = fig.add_gridspec(rows, cols, left=0.3, right=0.85, hspace=0.2, wspace=0.4)


        for i in range(startindex,index_max):  
            name = param_list[i][0]
            label = param_list[i][-1]
            if not name:
                return
            
            ax_additional = fig.add_subplot(gs[i%rows, i//rows])  
            if "redraw" in name.lower():
                button = Button(ax_additional, label)
                self.button = button    
            
            elif name!="":
                if isinstance(param_list[i][1],bool):
                    name, default,_,_,label = param_list[i]
                    check = CheckButtons(ax_additional, [label], [default])
                    self.parameter_dict[name] = check
                else:
                    name, initval, minval,maxval,label = param_list[i]
                    slider = Slider(ax_additional,label, minval, maxval, valinit=initval)
                    self.parameter_dict[name] = slider


        assert self.button #throws error if not defined, add REDRAW_PARAM somewhere in the return of get_parameter_list

             
    
    def append_redraw(self,redraw):
        self.button.on_clicked(redraw)

    def view_geometry(self):
        fig, ax = plt.subplots(figsize=(9, 8))
        def update(event):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.clear()
            paint_shapes_in_axis(self.get_shape_list(),ax,xlim,ylim,self.raster_count())
            ax.set_title(self.get_title())
            ax.figure.canvas.draw_idle()

        fig_sliders = plt.figure(figsize=(6, 8))
        self.create_sliders(fig_sliders)
        self.append_redraw(update)

        paint_shapes_in_axis(self.get_shape_list(),ax,self.get_init_xlim(),self.get_init_ylim(),self.raster_count())
        ax.set_title(self.get_title())

        plt.show()

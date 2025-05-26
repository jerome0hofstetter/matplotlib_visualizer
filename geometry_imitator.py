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

EMPTY_PARAM = ("",0,0,0,"")
REDRAW_PARAM = ("redraw",0,0,0,"Redraw")
SHOW_LEGEND = ("legend",True,0,0,"show legend")
RASTER_NUM_PARAM = ("raster",150,30,1000,"Raster Number")
class GeometryImitator(ABC):
    """
    extend this class, implement the abstract methods

    get an instance of it, call create_sliders, and append_redraw
    or directly use view_geometry

    when extending needs to define 
    - get_parameter_list
    - get_shape_list

    can define
    - annotations
    - get_geometry_values

    to use direcftly instantiate and call view_geometry
    
    """

    def __init__(self, view_size,see_negative_x=False,see_negative_y=False):
        self.view_size = view_size
        self.see_negative_y = see_negative_y
        self.see_negative_x = see_negative_x
        self.button = None
    
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
            result += f"\n{name:<20}: {val:>10.4f} {postfix:<6}"
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
        
        initialvalue being a number gives a slider, it being a boolean a checkbox
        """
        return [("a",1,1,2,"label")]
    
    @abstractmethod
    def get_shape_list(self) -> list[Shape]:
        pass
    

    def get_slider_dict(self):
        return self.parameter_dict
    
    def param_label(self,key):
        return [p[-1] for p in  self.get_parameter_list() if p[0]==key][0]

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

    def annotations(self,ax):
        """
        gets called after the shapes been drawn
        """
        pass

    def paint_geometry(self,ax,xlim,ylim ):
        x = np.linspace(*xlim, self.raster_count())
        y = np.linspace(*ylim, self.raster_count())

        X, Y = np.meshgrid(x, y)
        ax.set_aspect('equal') 
        ax.set_autoscale_on(False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        shapelist = self.get_shape_list()
        for shape in shapelist:
            if shape:
                shape.paint_shape(X,Y,ax)
                shape.after_drawing_shape(ax)
        legend_patches = [
            shape.get_label_patch() for shape in shapelist if shape
        ]

        self.annotations(ax)
        SHOW_LEGEND
        # Add legend
        if not SHOW_LEGEND in self.get_parameter_list() or self["legend"]:
            h, l = ax.get_legend_handles_labels()
            ax.legend(handles=legend_patches+h, title="Volumes")

    def view_geometry(self):
        fig, ax = plt.subplots(figsize=(9, 8))
        def update(event):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.clear()
            self.paint_geometry(ax,xlim,ylim)
            ax.set_title(self.get_title())
            ax.figure.canvas.draw_idle()

        fig_sliders = plt.figure(figsize=(6, 8))
        self.create_sliders(fig_sliders)
        self.append_redraw(update)
        self.paint_geometry(ax,self.get_init_xlim(),self.get_init_ylim())
        ax.set_title(self.get_title())

        plt.show()

    def start_end_segment_annotation(self,ax,ref,vec,keys,is_start = True,reference_scaler=1):
        """
        given axis, ref point (either start or endpoint , depending on is_start), a direction (vector that will get normed)
        and a list of keys, draws a series in line described by start,vec with lengths taken from the keys
        """
        ref = np.array(ref)
        vec = vec/np.linalg.norm(vec)
        start = ref if is_start else ref - vec * np.sum([self[key] for key in keys])
        for key in keys:
            end = start + vec*self[key]
            draw_annotation_arrow(ax,start,end,self.param_label(key),reference_scaler=reference_scaler)
            start = end
def draw_annotation_arrow(ax,start,end,label,minimum_font = 2,reference_scaler=1,color="red",label_color = "orange", base_fontsize=20,head_width_ratio = 0.05, head_length_ratio = 0.1):
    start = np.array(start)
    end= np.array(end)
    vec = end - start
    length = np.linalg.norm(vec)
    ax.arrow(*start, *vec, head_width=length*head_width_ratio, head_length=length*head_length_ratio, 
         fc=color, ec=color, length_includes_head=True)

    yleft,yright = ax.get_ylim()
    reference_size = yright-yleft
    #fontsize scaled based on relative arrowsize with minimum fontsize
    fontsize = max(base_fontsize * length/reference_size * reference_scaler,minimum_font)
    if fontsize <=40:
        #add label in  the center of arrow
        midpoint = (start + end) / 2
        ax.text(*midpoint, label, fontsize=fontsize, ha='center', va='center',color=label_color)

def get_unit_vector(degree):
    """
    returns unitvector with angle in degrees from x axis 
    """
    alpha = np.radians(degree)
    return np.array([np.cos(alpha), np.sin(alpha)])
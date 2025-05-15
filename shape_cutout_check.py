from geometry_imitator import *

"""
possible geometries for a calorimeter
"""

class CaloShape4PointShapeCheck(GeometryImitator):
    def __init__(self):
        self.view_size = 6
        self.see_negative_y = False
        self.see_negative_x = False
        self.button = None

    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",1,0.2,3,"inner radius"),
            ("rOut",3,2,5,"outer radius"),
            ("angle", 45,0,180,"angle"),
            ("xcutoff",3,0,5,"xcutoff"),
            ("ycutoff",0,0,2,"ycutoff"),
            ("offset",0,0,3,"offset"),
            REDRAW_PARAM,
            RASTER_NUM_PARAM
        ]
        return list

    
    def get_shape_list(self) -> list[Shape]:
        calo = CaloShape4PointDirect("blue","caloshape",self["rIn"],self["rOut"], self["xcutoff"],self["ycutoff"],self["angle"], self["offset"])
        cIn,cOut = calo.get_cone_points()
        ref_angle = np.degrees(np.arctan2(cOut[1],cOut[0]))
        disk = CaloShape4PointDirect((0.082, 0.929, 0.8, 0.5),"hollow sphere",self["rIn"],self["rOut"],self["rOut"]*2,0,ref_angle,0)
        return [calo,disk,*calo.get_cutouts()]
    
def get_point(r,angle):
        ang = np.radians(angle)
        return (r*np.cos(ang), r* np.sin(ang))
    
class DiskWithStopShapeCheck(GeometryImitator):
    def __init__(self):
        self.view_size = 5
        self.see_negative_y = False
        self.see_negative_x = False
        self.button = None

    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",2.8,0,3,"inner radius"),
            ("rOut",3,3,5,"outer radius"),
            ("angleIn", 45,0,180,"angleIn"),
            ("angleOut", 45,0,180,"angleOut"),
            REDRAW_PARAM,
            RASTER_NUM_PARAM
        ]
        return list

    
    def get_shape_list(self) -> list[Shape]:
        stoppers = (  get_point(self["rIn"],self["angleIn"]), get_point(self["rOut"],self["angleOut"]) )
        
        disk = DiskWithStopShape("blue","disk with stop", self["rIn"], self["rOut"], stoppers)
        shapes = [disk,]
        shapes.extend(disk.get_cutout_shapes())
        return shapes
    
class PolygonSectioningCheck(GeometryImitator):
    def __init__(self):
        self.view_size = 10
        self.see_negative_y = False
        self.see_negative_x = False
        self.button = None

    def annotations(self, ax):
        try:
            poly = self.get_shape_list()[0]
            edges = poly.edges
            sectioning = get_sectioning_of_simple_polygon(edges)
            for x,(y_min,y_max) in sectioning:
                if y_min == y_max:
                    ax.plot(x,y_min,color='red',marker='o', markersize=5)
                ax.vlines(x,y_min,y_max, color='red', linestyle='--') 
        except Exception as e:
            print(f"An error occurred: {e}")


    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:

        coor = ["x","y"]
        def default_val(coor,number):
            if coor ==0: #x value
                return number//2* 4 + 2
            else:
                return (number +1 )//2 % 2 * 4 + 2
        paramList = [ (f"{key}{num}",default_val(key,num) ,0,10,  f"{coor[key]}{num}")  for num in range(4)  for key in range(2)     ]
        paramList.extend([REDRAW_PARAM,
            RASTER_NUM_PARAM])

        return paramList

    
    def get_shape_list(self) -> list[Shape]:
        poly = PolygonShape((0.082, 0.929, 0.8, 0.38),"polygon", [(self[f"0{num}"],self[f"1{num}"]) for num in range(4)])
        return [poly,]
    

geometryList = [DiskWithStopShapeCheck(), PolygonSectioningCheck(),CaloShape4PointShapeCheck()]
for geometry in geometryList:
    geometry.view_geometry()
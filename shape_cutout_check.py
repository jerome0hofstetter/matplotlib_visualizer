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
                if number==4 :
                    return 4
                return number//2* 4 + 2
            else:
                if number==4 :
                    return 0
                return (number +1 )//2 % 2 * 4 + 2
        paramList = [ (f"{key}{num}",default_val(key,num) ,0,10,  f"{coor[key]}{num}")  for num in range(5)  for key in range(2)     ]
        paramList.extend([REDRAW_PARAM,
            RASTER_NUM_PARAM])

        return paramList

    
    def get_shape_list(self) -> list[Shape]:
        poly = PolygonShape((0.082, 0.929, 0.8, 0.38),"polygon", [(self[f"0{num}"],self[f"1{num}"]) for num in range(5)])
        return [poly,]
    


geometryList = [DiskWithStopShapeCheck(),CaloShape4PointShapeCheck(),PolygonSectioningCheck()]
desc_list = [
    "this shape is parameterised by inner and outer radius with a point on each radius being the stop \n"+
    "this shows how this shape can be created via a makesphere (hollow circle stopping at some angle) and then a cutout via a polygon suitably placed \n"+
    "the stoppers are for ease of use parameteres by an angle, can throw an error if the line connecting the stoppers intersect with the inner radius, this working properly can be verified with this",

    "this shape is parameterised more heavily, inner/outer radius, xcutoff,ycutoff and another line parallel to a line through zero at given angle \n"+
    "but with a given offset. This shows how to create this shape, again with a makesphere like shape and then subtracting away via polygons \n",
    "multiple (if needed) cutouts with edges going a bit further than necessary are used to ensure that no points are left where bounday becomes hard to evaluate",

    "if a polygon cant be parameterised via edges but instead via sections of the form (x,(y_min,y_max)) a general function making this parameter change is handy \n"+
    "this is a visual test if the function handles it well by showing the sections and throwing errors if not sectionable or has self intersections"
]
for i,geometry in enumerate(geometryList):
    print()
    print(desc_list[i])
    print("close both windows to go to the next geometry")
    geometry.view_geometry()
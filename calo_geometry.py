
import numpy as np
from geometry_imitator import *

"""
possible geometries for a calorimeter
"""

# this was more of an initial testrun as well as the base of the next
class Calorimeter(GeometryImitator):
    MAX_R = 120
    def __init__(self):
        self.view_size = self.MAX_R
        self.see_negative_y = True
        self.see_negative_x = True
        self.button = None

    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",15,5,30,"inner radius"),#
            ("active",54.568,30,70,"active thickness"),#
            ("cutoff",42,10,90,"x cutoff"),#
            ("alpha",35,0,90,"opening angle"),#

            ("innerWindow",0.05,0,0.5,"inner window"),#
            ("innerConeA",2,0,2,"inner cone A"),#
            ("innerConeB",2,0,3,"inner cone B"),#
            ("innercut",2,0,3,"inner cutoff"),#
            ("innerRshell",2,0,3,"inner shell"),#

            ("outerWindow",0.02,0,0.5,"outer window"),#
            ("outerConeA",2,0,2,"outer cone A"),#
            ("outerConeB",2,0,3,"outer cone B"),#
            ("outercut",2,0,3,"outer cutoff"),#
            ("outerRshell",2,0,3,"outer shell"),#

            ("innerRVacuum",2,0,4,"inner vacuum"),#
            ("outerRVacuum",2,0,4,"outer vacuum"),#
            ("coneVacuumA",2,0,4,"cone vacuum A"),#
            ("coneVacuumB",2,0,4,"cone vacuum B"),#
            ("cutoffVacuum",2,0,4,"cutoff vacuum"),#

            ("innerRsensor",0.06,0,0.3,"inner sensorR"),#
            ("outerRsensor",2,0,3,"outer sensorR"),#
            ("sensorCut",0.06,0,0.3,"cone cutoff"),#
            ("sensorConeA",0.06,0,0.3,"cone sensor A"),#
            ("sensorConeB",0.06,0,0.3,"cone sensor B"),#

            REDRAW_PARAM,
            RASTER_NUM_PARAM,
            ("innerwCutoff",12.3,0,25,"inner window cutoff"),#
            ("outerwCutoff",11.20,0,25,"outer window cutoff"),#
        ]
        return list

    def inner_radius(self):
        return self["rIn"]
    def inner_vacuum_d(self):
        return self["innerVacuum"]
    def inner_window_rad(self):
        return self.inner_radius() + self["innerWindow"]
    def outer_window_in_rad(self):
        return self.inner_window_rad() + self["innerRVacuum"]
    def inner_sensor_start(self):
        return self.outer_window_in_rad() + self["outerWindow"]
    def active_inner_rad(self):
        return self.inner_sensor_start() + self["innerRsensor"]
    def active_outer_rad(self):
        return self.active_inner_rad() + self["active"]
    def outer_sensor_end(self):
        return self.active_outer_rad() + self["outerRsensor"]
    def outer_vacuum_start(self):
        return self.outer_sensor_end() + self["innerRshell"]
    def outer_vacuum_end(self):
        return self.outer_vacuum_start() + self["outerRVacuum"]
    def outer_radius(self):
        return self.outer_vacuum_end() + self["outerRshell"]
    def outer_cone_offsetA(self):
        return self["outerConeA"]
    def cone_vacuum_end_offsetA(self):
        return self.outer_cone_offsetA() + self["coneVacuumA"]
    def inner_shell_cone_offsetA(self):
        return self.cone_vacuum_end_offsetA() + self["innerConeA"]
    def active_start_offsetA(self):
        return self.inner_shell_cone_offsetA() + self["sensorConeA"]
    def outer_cone_offsetB(self):
        return self["outerConeA"]
    def cone_vacuum_end_offsetB(self):
        return self.outer_cone_offsetB() + self["coneVacuumA"]
    def inner_shell_cone_offsetB(self):
        return self.cone_vacuum_end_offsetB() + self["innerConeA"]
    def active_start_offsetB(self):
        return self.inner_shell_cone_offsetB() + self["sensorConeA"]
    def shell_cutoff(self):
        return self["cutoff"]
    def vacuum_start_cutoff(self):
        return self.shell_cutoff() - self["outercut"]
    def vacuum_end_cutoff(self):
        return self.vacuum_start_cutoff() - self["cutoffVacuum"]
    def inner_shell_end_cutoff(self):
        return self.vacuum_end_cutoff() - self["innercut"]
    def active_cutoff(self):
        return self.inner_shell_end_cutoff() - self["sensorCut"]
    
    def get_inner_window_x_cutoff(self):
        return min(self["innerwCutoff"],self["cutoff"])
    
    def get_outer_window_x_cutoff(self):
        return min(self["outerwCutoff"],self["cutoff"])

    def get_active_shape(self):
        return CaloShape("violet","active volume",self.active_inner_rad(),self.active_outer_rad(),
                         self["alpha"],self.active_start_offsetA(), self.active_start_offsetB(),self.active_cutoff() )
    
    def get_sensor_shape_pre(self):
        return CaloShape("pink","sensor volume",self.inner_sensor_start(),self.outer_sensor_end(),
                         self["alpha"],self.inner_shell_cone_offsetA(), self.inner_shell_cone_offsetB(),self.inner_shell_end_cutoff() )
    
    def get_sensor_shape(self):
        return CombiShape("pink","sensor volume", (self.get_sensor_shape_pre(),), (self.get_active_shape(),))
    
    def get_inner_shell_pre(self):
        return CaloShape("grey","inner shell",self.outer_window_in_rad(),self.outer_vacuum_start(),
                         self["alpha"],self.cone_vacuum_end_offsetA(), self.cone_vacuum_end_offsetB(),self.vacuum_end_cutoff() )
    
    def get_inner_shell(self):
        return CombiShape("grey","inner shell", (self.get_inner_shell_pre(),), (self.get_sensor_shape_pre(),))
    
    def get_vacuum_vol_pre(self):
        return CaloShape("yellow","vacuum",self.inner_window_rad(),self.outer_vacuum_end(),
                         self["alpha"],self.outer_cone_offsetA(), self.outer_cone_offsetB(),self.vacuum_start_cutoff() )
    
    def get_vacuum_vol(self):
        return CombiShape("yellow","vacuum", (self.get_vacuum_vol_pre(),), (self.get_inner_shell_pre(),))
    
    def get_outer_shell_pre(self):
        return CaloShape("black","outer shell",self.inner_radius(),self.outer_radius(),
                         self["alpha"],0,0,self.shell_cutoff() )
    
    def get_outer_shell(self):
        return CombiShape("black","outer shell", (self.get_outer_shell_pre(),), (self.get_vacuum_vol_pre(),))
    
    def get_inner_window(self):
        return CaloShape("lightblue","inner window",self.inner_radius(),self.inner_window_rad(),
                         self["alpha"],0,0,self.get_inner_window_x_cutoff() )
    def get_outer_window(self):
        return CaloShape("darkblue","outer window",self.outer_window_in_rad(),self.inner_sensor_start(),
                         self["alpha"],0,0,self.get_outer_window_x_cutoff() )

    def get_geometry_values(self):
        innerpoint = self.get_active_shape().innerpoint
        outerpoint = self.get_active_shape().outerpoint
        return [
            ("innerpoint active angle",np.degrees(np.arctan2(innerpoint[1],innerpoint[0]) ),"degrees"),
            ("outerpoint active angle",np.degrees(np.arctan2(outerpoint[1],outerpoint[0]) ),"degrees")
        ]


    def get_shape_list(self) -> list[Shape]:
        return [
            self.get_active_shape(),
            self.get_sensor_shape(),
            self.get_inner_shell(),
            self.get_vacuum_vol(),
            self.get_outer_shell(),
            self.get_inner_window(),
            self.get_outer_window()
        ]
    

class CalorimeterCurrentMimick(Calorimeter):
    MAX_R = 120

    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",15,5,30,"inner radius"),#
            ("active",54.568,30,70,"active thickness"),#
            ("cutoff",42,10,90,"x cutoff"),#
            ("alpha",35,0,90,"opening angle"),#

            ("innerWindow",1,0,0.5,"inner window"),#
            ("innerConeA",2,0,2,"inner cone A"),#
            ("innerConeB",2,0,3,"inner cone B"),#
            ("innercut",2,0,3,"inner cutoff"),#
            ("innerRshell",2,0,3,"inner shell"),#

            ("outerWindow",1,0,0.5,"outer window"),#
            ("outerConeA",2,0,2,"outer cone A"),#
            ("outerConeB",2,0,3,"outer cone B"),#
            ("outercut",2,0,3,"outer cutoff"),#
            ("outerRshell",2,0,3,"outer shell"),#

            ("innerRVacuum",2,0,4,"inner vacuum"),#
            ("outerRVacuum",2,0,4,"outer vacuum"),#
            ("coneVacuumA",2,0,4,"cone vacuum A"),#
            ("coneVacuumB",2,0,4,"cone vacuum B"),#
            ("cutoffVacuum",2,0,4,"cutoff vacuum"),#

            ("innerRsensor",0.06,0,0.3,"inner sensorR"),#
            ("outerRsensor",2,0,3,"outer sensorR"),#
            ("sensorCut",0.06,0,0.3,"cone cutoff"),#
            ("sensorConeA",0.06,0,0.3,"cone sensor A"),#
            ("sensorConeB",0.06,0,0.3,"cone sensor B"),#

            ("outer_cutoff_xtrans",0,0,12,"outer window \ncutoff translation"),

            REDRAW_PARAM,
            RASTER_NUM_PARAM
        ]
        return list
    
    def _s_a(self):
        return np.sin(np.radians(self["alpha"]))
    def _c_a(self):
        return np.cos(np.radians(self["alpha"]))
    
    def get_inner_window_x_cutoff(self):
        return self.get_cone_xstart()
    
    def get_outer_window_x_cutoff(self):
        return self.get_cone_xstart()+self["outer_cutoff_xtrans"]
    
    def get_cone_xstart(self):
        return self._c_a()*self.inner_radius()
    def get_cone_xend(self):
        return self._c_a()*self.outer_radius()
    
    def get_outer_cone_starty_A(self):
        return self._s_a()*self.inner_radius()
    
    def get_outer_cone_starty_B(self):
        return self._s_a()*self.outer_radius()
    
    def get_outer_cone_endy_A(self):
        return self.get_outer_cone_starty_A() + self["outerConeA"]*self._s_a()
    
    def get_outer_cone_endy_B(self):
        return self.get_outer_cone_starty_B() + self["outerConeB"]*self._s_a()
    
    def get_inner_cone_starty_A(self):
        return self.get_outer_cone_endy_A() + self["coneVacuumA"]/self._c_a()
    
    def get_inner_cone_starty_B(self):
        return self.get_outer_cone_endy_B() + self["coneVacuumB"]/self._c_a()
    
    def get_inner_cone_endy_A(self):
        return self.get_inner_cone_starty_A() + self["innerConeA"]*self._s_a()
    
    def get_inner_cone_endy_B(self):
        return self.get_inner_cone_starty_B() + self["innerConeB"]*self._s_a()
    

    
    def get_active_shape(self):
        return CaloShape("violet","active volume",self.active_inner_rad(),self.active_outer_rad(),
                         self["alpha"],self.active_start_offsetA(), self.active_start_offsetB(),self.active_cutoff() )
    
    def get_sensor_shape_pre(self):
        return CaloShape("pink","sensor volume",self.inner_sensor_start(),self.outer_sensor_end(),
                         self["alpha"],self.inner_shell_cone_offsetA(), self.inner_shell_cone_offsetB(),self.inner_shell_end_cutoff() )
    
    def get_sensor_shape(self):
        return CombiShape("pink","sensor volume", (self.get_sensor_shape_pre(),), (self.get_active_shape(),))
    
    
    def get_vacuum_vol_pre(self):
        return CaloShape("yellow","vacuum",self.inner_radius(),self.outer_radius(),
                         self["alpha"],0,0,self.shell_cutoff() )
    
    def get_vacuum_vol(self):
        return CombiShape("yellow","vacuum", (self.get_vacuum_vol_pre(),), (self.get_sensor_shape_pre(),))
    
    def get_outer_cone_pre(self):
        edges = [
            (self.get_cone_xstart(),self.get_outer_cone_starty_A()),
            (self.get_cone_xstart(),self.get_outer_cone_endy_A()),
            (self.get_cone_xend(),self.get_outer_cone_endy_B()),
            (self.get_cone_xend(),self.get_outer_cone_starty_B()),

        ]
        return MirrorPolygonShape("0.8","outer cone",edges)

    def get_outer_cone(self):
        return IntersectionShape("0.8","outer cone",self.get_outer_cone_pre(),self.get_vacuum_vol_pre())
    
    def get_inner_cone_pre(self):
        edges = [
            (self.get_cone_xstart(),self.get_inner_cone_starty_A()),
            (self.get_cone_xstart(),self.get_inner_cone_endy_A()),
            (self.get_cone_xend(),self.get_inner_cone_endy_B()),
            (self.get_cone_xend(),self.get_inner_cone_starty_B()),

        ]
        return MirrorPolygonShape("0.6","inner cone",edges)

    def get_inner_cone(self):
        return IntersectionShape("0.6","inner cone",self.get_inner_cone_pre(),self.get_vacuum_vol_pre())
    
    
    def get_shape_list(self) -> list[Shape]:
        return [
            self.get_active_shape(),
            self.get_sensor_shape(),
            self.get_vacuum_vol(),
            self.get_outer_cone(),
            self.get_inner_cone(),
            self.get_inner_window(),
            self.get_outer_window()
        ]
    

def get_outer_radial_point(calo_shape:CaloShape4PointDirect):
    """
    returns outer radius and the endpoint at the outer radius
    """
    return calo_shape.rOut, calo_shape.get_radius_points()[1]
def get_inner_radial_cone_points(calo_shape:CaloShape4PointDirect):
    """
    returns inner radius and the endpoint at the inner radius, innercone,outercone points
    """
    return calo_shape.rIn, calo_shape.get_radius_points()[0],*(calo_shape.get_cone_points())

def get_transistion_cone_points(calo_shape_outer,calo_shape_inner):
    """
    given an outer and inner calo shape (such that inner is contained in outer volumewise)

    returns 3 tuples, inner radius (rOuter,rInner) ,edges for transistion shape with the first two values being the inner radius points (pOuter,pInner)
    and lastly the edges for the coneshape
    """
    rIn,p1,c1,d1 = get_inner_radial_cone_points(calo_shape_outer)
    rOut,p2,c2,d2 = get_inner_radial_cone_points(calo_shape_inner)
    assert c2[0]<=c1[0] , "require outershape innercone point to be farther away from y axis as innershape innercone point"
    assert p1[0]<=c2[0] , "require outershape innercone point to be farther away from y axis as innershape inner radius point, perhaps distance between the shapes at the ycutoff was choosen too big"
    #this point is on the line formed by p1 and c1 at the same x as c2
    e1 = (c2[0],c1[1])
    return (rIn,rOut), [p1,p2,c2,e1], [e1,c2,d2,d1,c1]

def get_cutoff_wall_points(calo_shape_outer,calo_shape_inner):
    """
    returns the list of edges for the polygon formed by the x cutoff wall
    """
    c1,r1 = calo_shape_outer.get_cone_points()[1],calo_shape_outer.get_radius_points()[1]
    c2,r2 = calo_shape_inner.get_cone_points()[1],calo_shape_inner.get_radius_points()[1]
    return [c2,r2,r1,c1]


class CalorimeterUpdated(GeometryImitator):
    MAX_R = 120
    def __init__(self):
        self.view_size = self.MAX_R
        self.see_negative_y = True
        self.see_negative_x = True
        self.button = None

    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",15,5,30,"inner radius"),#
            ("active",54.568,30,70,"active thickness"),#
            ("cutoff",42,10,90,"x cutoff"),#
            ("alpha",35,0,90,"opening angle"),#

            ("outerWindow",0.02,0,0.5,"outer window"),#
            ("outerWindowTrW",0.3,0,0.8,"outer transition width"),#
            ("outerWindowTrH",0.02,0,0.4,"outer transition height"),#

            ("innerCone",2,0,2,"inner cone"),#
            ("innercut",2,0,3,"inner cutoff"),#
            ("innerRshell",2,0,3,"inner shell"),#

            ("innerWindow",0.05,0,0.5,"inner window"),#

            ("innerWindowTrW",0.3,0,0.8,"inner transition width"),#
            ("innerWindowTrH",0.02,0,0.4,"inner transition height"),#

            ("outerCone",2,0,2,"outer cone"),#
            ("outercut",2,0,3,"outer cutoff"),#
            ("outerRshell",2,0,3,"outer shell"),#

            ("innerRVacuum",2,0,4,"inner vacuum"),#
            ("outerRVacuum",2,0,4,"outer vacuum"),#
            ("coneVacuum",2,0,4,"cone vacuum"),#
            ("cutoffVacuum",2,0,4,"cutoff vacuum"),#

            ("innerRsensor",0.06,0,0.3,"inner sensorR"),#
            ("outerRsensor",2,0,3,"outer sensorR"),#
            ("sensorCut",0,0,0.3,"sensor cutoff"),#
            ("sensorCone",0.06,0,0.3,"cone sensor"),#
            

            ("detailed",False,0,0,"show detailed"),
            ("annotate",True,0,0,"show annotations"),
            ("arrows",True,0,0,"show arrow annotations"),
            ("arrowlabelscale",1,0,3,"label scaler"),
            SHOW_LEGEND,
            REDRAW_PARAM,
            RASTER_NUM_PARAM,
        ]
        return list
### ---------------- direct parameters
    def inner_radius(self):
        return self["rIn"]
    def inner_window_rad(self):
        return self.inner_radius() + self["innerWindow"]
    def outer_window_in_rad(self):
        return self.inner_window_rad() + self["innerRVacuum"]
    def inner_sensor_start(self):
        return self.outer_window_in_rad() + self["outerWindow"]
    def active_inner_rad(self):
        return self.inner_sensor_start() + self["innerRsensor"]
    def active_outer_rad(self):
        return self.active_inner_rad() + self["active"]
    def outer_sensor_end(self):
        return self.active_outer_rad() + self["outerRsensor"]
    def outer_vacuum_start(self):
        return self.outer_sensor_end() + self["innerRshell"]
    def outer_vacuum_end(self):
        return self.outer_vacuum_start() + self["outerRVacuum"]
    def outer_radius(self):
        return self.outer_vacuum_end() + self["outerRshell"]
    

    def outer_cone_offset(self):
        return self["outerCone"]
    def cone_vacuum_end_offset(self):
        return self.outer_cone_offset() + self["coneVacuum"]
    def inner_shell_cone_offset(self):
        return self.cone_vacuum_end_offset() + self["innerCone"]
    def active_start_offset(self):
        return self.inner_shell_cone_offset() + self["sensorCone"]
    
    def shell_cutoff(self):
        return self["cutoff"]
    def vacuum_start_cutoff(self):
        return self.shell_cutoff() - self["outercut"]
    def vacuum_end_cutoff(self):
        return self.vacuum_start_cutoff() - self["cutoffVacuum"]
    def inner_shell_end_cutoff(self):
        return self.vacuum_end_cutoff() - self["innercut"]
    def active_cutoff(self):
        return self.inner_shell_end_cutoff() - self["sensorCut"]

### -------------- parameters from ycutoff 
    def outer_window_upper_y_cutoff(self):
        sensor_pre = self.get_sensor_shape_pre()
        return sensor_pre.y_cutoff 
    def outer_window_lower_y_cutoff(self):
        return self.outer_window_upper_y_cutoff() - self["outerWindowTrH"]   
    def inner_window_upper_y_cutoff(self):
        vacuum_pre = self.get_vacuum_boundary_shape_pre()
        return vacuum_pre.y_cutoff    
    def inner_window_lower_y_cutoff(self):
        return self.inner_window_upper_y_cutoff() - self["innerWindowTrH"]
    
### more complex points

### ---- shapes
    def get_sensor_shape_pre(self): #
        return CaloShape4Point("pink","sensor volume",self.inner_sensor_start(),self.outer_sensor_end(),self.inner_shell_end_cutoff(),
                self["outerWindowTrW"], self["alpha"], self.inner_shell_cone_offset() )
    def get_active_shape(self): #
        return CaloShape4PointDirect("violet","active volume",self.active_inner_rad(),self.active_outer_rad(),self.active_cutoff(),
                self.outer_window_upper_y_cutoff(), self["alpha"], self.active_start_offset() )
    def get_vacuum_boundary_shape_pre(self): #
        return CaloShape4Point("pink","sensor volume",self.inner_window_rad(),self.outer_vacuum_end(),self.vacuum_start_cutoff(),
                self["innerWindowTrW"], self["alpha"], self.outer_cone_offset() )
    def get_outer_shell_pre(self): #
        return CaloShape4PointDirect("black","outer shell",self.inner_radius(),self.outer_radius(),
                self.shell_cutoff() , self.inner_window_lower_y_cutoff(), self["alpha"],0)
    def get_inner_shell_pre(self): #
        return CaloShape4PointDirect("grey","inner shell",self.outer_window_in_rad(),self.outer_vacuum_start(),
                self.vacuum_end_cutoff(),  self.outer_window_lower_y_cutoff() ,self["alpha"],self.cone_vacuum_end_offset() )
    

    def get_inner_transition(self):
        r,transistion,cone = get_transistion_cone_points(self.get_outer_shell_pre(),self.get_vacuum_boundary_shape_pre())
        color = (0.859, 0.624, 0.471,1) # orange like
        return MirrorPolygonShape(color,"inner transition",transistion)

    def get_outer_transition(self):
        r,transistion,cone = get_transistion_cone_points(self.get_inner_shell_pre(),self.get_sensor_shape_pre())
        color = (0.749, 0.345, 0.075,1) # deeper orange
        return MirrorPolygonShape(color,"outer transition",transistion)
    
    def get_outer_cone(self):
        r,transistion,cone = get_transistion_cone_points(self.get_outer_shell_pre(),self.get_vacuum_boundary_shape_pre())
        color = (0.839, 0.153, 0.827,1) # pink violet ish
        return MirrorPolygonShape(color,"outer cone",cone)
    
    def get_outer_cutoff(self):
        edges= get_cutoff_wall_points(self.get_outer_shell_pre(),self.get_vacuum_boundary_shape_pre())
        color = "black" 
        return MirrorPolygonShape(color,"outer cutoff",edges)
    def get_inner_cutoff(self):
        edges= get_cutoff_wall_points(self.get_inner_shell_pre(),self.get_sensor_shape_pre())
        color = "grey" 
        return MirrorPolygonShape(color,"inner cutoff",edges)

    def get_inner_cone(self):
        r,transistion,cone = get_transistion_cone_points(self.get_inner_shell_pre(),self.get_sensor_shape_pre())
        color = (0.741, 0.063, 0.514,1) # deeper pink
        return MirrorPolygonShape(color,"inner cone",cone)

    def get_outer_radial_shell(self):
        rIn,p1 = get_outer_radial_point(self.get_vacuum_boundary_shape_pre())
        rOut,p2 = get_outer_radial_point(self.get_outer_shell_pre())
        color = (0.373, 0.49, 0.353,1) # darkish green
        return DiskWithStopShape(color,"outer radial",rIn,rOut,(p1,p2)) 
    
    def get_inner_radial_shell(self):
        rIn,p1 = get_outer_radial_point(self.get_sensor_shape_pre())
        rOut,p2 = get_outer_radial_point(self.get_inner_shell_pre())
        color = (0.502, 0.678, 0.659,1) # darkish cyan
        return DiskWithStopShape(color,"inner radial",rIn,rOut,(p1,p2)) 
    
    def get_inner_window(self):
        r,transistion,cone = get_transistion_cone_points(self.get_outer_shell_pre(),self.get_vacuum_boundary_shape_pre())
        color = (0.063, 0.737, 1,1) # sky blue
        return DiskWithStopShape(color,"inner window",*r,transistion[:2]) 
    
    def get_outer_window(self):
        r,transistion,cone = get_transistion_cone_points(self.get_inner_shell_pre(),self.get_sensor_shape_pre())
        color = (0.169, 0.545, 0.69,1) # darker sky blue
        return DiskWithStopShape(color,"outer window",*r,transistion[:2]) 

    def get_sensor_shape(self):
        return CombiShape("pink","sensor volume", (self.get_sensor_shape_pre(),), (self.get_active_shape(),))
    def get_inner_shell(self):
        return CombiShape("grey","inner shell", (self.get_inner_shell_pre(),), (self.get_sensor_shape_pre(),))
    def get_vacuum_vol(self):
        return CombiShape("yellow","vacuum", (self.get_vacuum_boundary_shape_pre(),), (self.get_inner_shell_pre(),))
    def get_outer_shell(self):
        return CombiShape("black","outer shell", (self.get_outer_shell_pre(),), (self.get_vacuum_boundary_shape_pre(),))
    
# ----extending from baseclass
    def points_of_interest(self):
        """
        returns a list of tuples of the form (point,color , label)
        """
        return [
            ( self.get_sensor_shape_pre().get_radius_points()[0] , (0.706, 0.78, 0.325, 0.651) ,"innerpoint"),
            ( self.get_active_shape().get_radius_points()[1] , (0.408, 0.859, 0.259, 0.651) ,"outerpoint"),
        ]

    
    def arrow_annotations(self,ax):
        radius_keys = [
            "rIn",
            "innerWindow",
            "innerRVacuum",
            "outerWindow",
            "innerRsensor",
            "active",
            "outerRsensor",
            "innerRshell",
            "outerRVacuum",
            "outerRshell"
        ]
        
        self.start_end_segment_annotation(ax,(0,0),get_unit_vector(80),radius_keys,reference_scaler=self["arrowlabelscale"])
        active_shape = self.get_active_shape()
        cones = [np.array(point) for point in active_shape.get_cone_points()]
        outerRadP = np.array(active_shape.get_radius_points()[1])
        cutoff_ref = (cones[1] + outerRadP)/2
        offset_ref = (cones[0] + cones[1])/2
        xcutoff_keys = [
            "outercut",
            "cutoffVacuum",
            "innercut",
            "sensorCut",
        ]
        self.start_end_segment_annotation(ax,cutoff_ref,(-1,0),xcutoff_keys,False, reference_scaler=self["arrowlabelscale"])
        offset_keys = [
            "outerCone",
            "coneVacuum",
            "innerCone",
            "sensorCone",
        ]
        self.start_end_segment_annotation(ax,offset_ref,get_unit_vector(90+self["alpha"]),offset_keys,False,reference_scaler=self["arrowlabelscale"])
        draw_annotation_arrow(ax,(0,0),(self["cutoff"],0), "x cutoff")

        # of form [p1,p2,c2,e1]
        inner_transistion_edges = self.get_inner_transition().edges
        draw_annotation_arrow(ax,inner_transistion_edges[2],inner_transistion_edges[1], "inner Window \n transition width",reference_scaler=0.6*self["arrowlabelscale"])
        draw_annotation_arrow(ax,inner_transistion_edges[2],inner_transistion_edges[3], "inner Window \n transition height",reference_scaler=7*self["arrowlabelscale"])
        draw_annotation_arrow(ax,(inner_transistion_edges[3][0],0),inner_transistion_edges[3], "zoom into inner window \n depends on inner window width&height",
                              reference_scaler=1*self["arrowlabelscale"],color="blue",label_color="black")
        outer_transistion_edges = self.get_outer_transition().edges
        draw_annotation_arrow(ax,outer_transistion_edges[2],outer_transistion_edges[1], "outer Window \n transition width",reference_scaler=0.6*self["arrowlabelscale"])
        draw_annotation_arrow(ax,outer_transistion_edges[2],outer_transistion_edges[3], "outer Window \n transition height",reference_scaler=7*self["arrowlabelscale"])
        draw_annotation_arrow(ax,(outer_transistion_edges[3][0],0),outer_transistion_edges[3], "zoom into outer window \n depends on outer window width&height",
                              reference_scaler=1*self["arrowlabelscale"],color="blue",label_color="black")
        

    def annotations(self,ax):
        if not self["annotate"] or not self["arrows"]:
            return
        for p,color,label in self.points_of_interest():
            ax.plot(*p, marker='o', markersize=10, color=color, label =label)
        if self["arrows"]:
            self.arrow_annotations(ax)



    def get_geometry_values(self):
        liqXe_density = 3.1 # g/ml   or kg/l  #TODO get correct value, varies with temperature and pressure
        active_vol_liter = self.get_active_shape().get_rotation_volume()/1000
        base =  [ 
            ("active volume",active_vol_liter,"liter"),
            ("active volume",active_vol_liter*liqXe_density,"kg"),
        ]
        if self["annotate"]:
            angles = [(f"angle of {label}",np.degrees(np.arctan2(y,x)),"degrees") for (x,y),color,label in self.points_of_interest()]
            base.extend(angles)
        return base


    def get_shape_list(self) -> list[Shape]:
        if self["detailed"] :
            return [
                self.get_vacuum_vol(),
                self.get_sensor_shape(),
                self.get_active_shape(),

                self.get_outer_radial_shell(),
                self.get_inner_radial_shell(),

                self.get_outer_cutoff(),
                self.get_inner_cutoff(),

                self.get_inner_window(),
                self.get_outer_window(),
                
                self.get_outer_cone(),
                self.get_inner_cone(),
                self.get_inner_transition(),
                self.get_outer_transition(),
            ]
        else:
            return [
                self.get_outer_shell(),
                self.get_vacuum_vol(),
                self.get_inner_shell(),
                self.get_sensor_shape(),
                self.get_active_shape(),
            ]


print("this visualises the liquid xenon geometry with all possible parameters forming the geometry itself")
print("conceptually has an outer layer, a vacuum, an inner layer, a sensor layer and in the center the active volume")
print("the shapes of the layers are all of type Calo4PointShape just with different parameters (and insides subtracted)")
print("if 'see detailed' is off those layers are shown which fully create the main geometry, \n while if on it shows the individuel parts (which now only depend on the layers and some of the distinct points of them)")
print("all shapes are either created via cuting out other shapes or baseshapes, baseshapes are the Calo4PointShape, DiskWithStopShape or polygons")
print()
print("considerations to shape")
print("the vacuum is needed to termally isolate the center (liquid xenon), the inner and outer shells to contain the liquid xenon and vacuum, sensor layer for sipms and pmts ")
print("inner and outer shell are sectioned (sensor should also be but not needed for this visualisation) as they should be individual volumes, which is easier for simulation analysis after")
print("the inner radial part of the inner and other shells are thin windows to reduce energy loss of particles coming from atar, which are then welded to the cone, that transiston is its own shape")
print()
print("other boolean options have control if certain things should be shown on top of the geometry,the legend, arrow annotation (they show how different parameter affect the geometry) or other annotations (like distincgt points)")

print("use the sliders to change parameters of the geometry above the boolean options and after changing or zooming in use redraw to redo the pixelated view")
geometry = CalorimeterUpdated()
geometry.view_geometry()
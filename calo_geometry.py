
import numpy as np
from geometry_imitator import *

"""
possible geometries for a calorimeter
"""

class Mockup(GeometryImitator):


    def get_parameter_list(self) -> list[tuple[str,float,float,float,str]]:
        list = [
            ("rIn",1,0.2,3,"inner radius"),
            ("rOut",3,2,5,"outer radius"),
            ("thetaIn", 45,0,180,"inner rad angle"),
            ("thetaOut", 45,0,180,"outer rad angle"),
            REDRAW_PARAM,
            RASTER_NUM_PARAM
        ]
        return list

    
    def get_shape_list(self) -> list[Shape]:
        calo = DiskWithStopShape("blue","testshape",self["rIn"],self["rOut"],(get_cirle_point(self["rIn"], self["thetaIn"]),get_cirle_point(self["rOut"],self["thetaOut"])))

        return [calo,]
    

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
            ("innerpoint active angle",f"{np.degrees(np.arctan2(innerpoint[1],innerpoint[0]) ):3.2f}","degrees"),
            ("outerpoint active angle",f"{np.degrees(np.arctan2(outerpoint[1],outerpoint[0]) ):3.2f}","degrees")
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
    #this point is on the line formed by p1 and c1 at the same x as c2
    e1 = (c2[0],c1[1])
    return (rIn,rOut), [p1,p2,c2,e1], [e1,c2,d2,d1,c1]

def get_cutoff_wall_points(calo_shape_outer,calo_shape_inner):
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
            REDRAW_PARAM,
            RASTER_NUM_PARAM,
        ]
        return list
### ---------------- direct parameters
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
    

    def get_geometry_values(self):
        return []


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
                self.get_inner_transition(),
                self.get_outer_window(),
                self.get_outer_transition(),
                self.get_outer_cone(),
                self.get_inner_cone()
            ]
        else:
            return [
                self.get_outer_shell(),
                self.get_vacuum_vol(),
                self.get_inner_shell(),
                self.get_sensor_shape(),
                self.get_active_shape(),
            ]
        


geometry = CalorimeterUpdated()
geometry.view_geometry()
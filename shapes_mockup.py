import numpy as np
from scipy.spatial.transform import Rotation as Rot
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#### --------------------------------
#defines 2d shapes very losely as just a class that can decide if a certain point is in the shape or not
#### --------------------------------

def get_circle_intersect_x(r,y):
    """
    given radius r of a circle centered at 0, height y, return positive x such that (x,y) is a point on the circle
    because of symmetry x,y can be changed
    """
    assert r>y , "failed as no such point on the circle exists"
    return np.sqrt(r**2-y**2)

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        print(p1,p2)
        raise ValueError("Cannot compute slope for vertical line (x1 == x2).")
    m = (y2 - y1) / (x2 - x1)
    a = y1 - m * x1
    return m, a


def calo_shape_parameter_extrapolate(parameter):
    innerpoint,outer_slope_point,outerpoint,angle = parameter
    r_in =  np.linalg.norm(innerpoint)
    r_out = np.linalg.norm(outerpoint)
    return r_in,r_out,angle

def calo_shape_check_vec(points,parameter):
    
    innerpoint,outer_slope_point,outerpoint,angle = parameter
    r_in,r_out,angle = calo_shape_parameter_extrapolate(parameter)
    r_p = np.linalg.norm(points,axis=1)
    m,a = line_from_points(innerpoint,outer_slope_point)
    
    is_in = np.ones_like(points[:,0], dtype=bool)
    # radius check + x_cutout (positive) check
    is_in = is_in & (r_in<r_p) & (r_out>r_p) & (points[:,0]<outer_slope_point[0])
    is_in = is_in & ((points[:,1]**2>(m *points[:,0]+a )**2) |(points[:,0]<0) )
    return is_in

def is_point_in_polygon(p:tuple[float,float], poly: list[tuple[float, float]]) -> bool:
    """
    Determine if the point is on the inside, corner, or boundary of the polygon defined via its edges
    (even-odd algorithm taken from https://en.wikipedia.org/wiki/Even–odd_rule#Implementation)
    Args:
      x -- The x coordinates of point.
      y -- The y coordinates of point.
      poly -- a list of tuples [(x, y), (x, y), ...]

    Returns:
      True if the point is in the path or is a corner or on the boundary"""
    x,y = p
    c = False
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[i - 1]
        if (x == ax) and (y == ay):
            # point is a corner
            return True
        if (ay > y) != (by > y):
            slope = (x - ax) * (by - ay) - (bx - ax) * (y - ay)
            if slope == 0:
                # point is on boundary
                return True
            if (slope < 0) != (by < ay):
                c = not c
    return c



def plot_halfcircle_from_point(r,point):
    angle = np.arctan2(point[1],point[0])
    theta = np.linspace(angle, np.pi, 100)
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    return x_circle,y_circle
from abc import ABC, abstractmethod
class Shape(ABC):
    """
    abstract class for a shape
    shapes need to either implement their own mesh_check or point_is_in function 


    """
    def point_is_in(self,point:tuple[float,float]):
        """
        given a point return if this point is inside the shape or not
        """
        raise NotImplementedError()

    def mesh_check(self,X,Y):
        """
        given X,Y obtained from numpy meshgrid return an array of true false if those points are in the shape or not
        
        uses should_include vectorised to check which points described in the mesh are inside it 
        """

        def should_include(x,y):
            return self.point_is_in((x,y))
        return np.vectorize(should_include)(X, Y)
    
    def should_points_be_marked(self):
        return False
    
    def after_drawing_shape(self,ax):
        if self.should_points_be_marked():
            self.mark_defining_points(ax)

    def mark_defining_points(self,ax):
        pass
    
    def get_color(self):
        return self.color
    
    def set_color(self,color):
        self.color = color
    
    def get_color_rgba(self):
        if isinstance(self.color,str):
            return to_rgba(self.color)
        else:
            return self.color

    def get_label_patch(self):
        return mpatches.Patch(color=self.get_color_rgba(), label=self.label) 

    def paint_shape(self,X,Y,ax = None):
        include_points = self.mesh_check(X,Y)

        color_val = self.get_color_rgba()
        rgba_image = np.zeros((*include_points.shape, 4))
        rgba_image[include_points] = color_val

        ax.imshow(rgba_image, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])

class PolygonShape(Shape):
    def __init__(self,color,label, edges : list[tuple[float,float]]):
        self.color = color
        self.label = label
        self.edges = edges

    def point_is_in(self,point):
        return is_point_in_polygon(point,self.edges)
    
class MirrorPolygonShape(PolygonShape):
    def point_is_in(self,point):
        x,y=point
        return super().point_is_in((x,np.abs(y)))
    
class CircleShape(Shape):
    """
    defines a circleshape with radius rOut, with a radial cutout of radius rIn and only from and to given theta angle in radians

    theta angle starts at 0 at +x axis and ends at -x axis at pi, mirrored over x axis
    """
    def __init__(self,color,label, rIn,rOut, theta_start = 0,theta_end = np.pi):
        self.color = color
        self.label = label
        self.rIn = rIn
        self.rOut = rOut
        self.theta_start = theta_start
        self.theta_end = theta_end

    def point_is_in(self,point):
        x,y  = point
        r_p = np.linalg.norm(point)
        angle_rad = np.arctan2(np.abs(y), x)
        return self.rIn<=r_p and r_p <= self.rOut and self.theta_start <= angle_rad and angle_rad <= self.theta_end
    

class CombiShape(Shape):
    def __init__(self,color,label, additionShapes,subtractionShapes):
        self.color = color
        self.additionShapes = additionShapes
        self.subtractionShapes = subtractionShapes
        self.label = label

    def mesh_check(self,X,Y):
        bool_matrix_add = [ shapes.mesh_check(X,Y)  for shapes in self.additionShapes]
        bool_matrix_sub = [ shapes.mesh_check(X,Y)  for shapes in self.subtractionShapes]
        final_add_mask = np.logical_or.reduce(bool_matrix_add)

        final_sub_mask = np.logical_or.reduce(bool_matrix_sub)

        return final_add_mask & ~final_sub_mask

class IntersectionShape(Shape):
    def __init__(self,color,label, A,B):
        self.color = color
        self.A = A
        self.B = B
        self.label = label

    def mesh_check(self,X,Y):
        return self.A.mesh_check(X,Y) & self.B.mesh_check(X,Y)
    

class CaloShapeDirect(Shape):
    def __init__(self, color,label,innerpoint,outer_slope_point,outerpoint):
        self.color = color
        self.label = label
        self.innerpoint,self.outer_slope_point,self.outerpoint = innerpoint,outer_slope_point,outerpoint
        self.rIn = np.linalg.norm(innerpoint)
        self.rOut = np.linalg.norm(outerpoint)
    
    def point_is_in(self,point):
        r_p = np.linalg.norm(point)
        m,a = line_from_points(self.innerpoint,self.outer_slope_point)
        return self.rIn<=r_p and r_p <= self.rOut and point[0]<=self.outer_slope_point[0] and (point[1]**2>=(m *point[0]+a )**2 or point[0]<0)
    
def get_intersection(v,a,target,index):
        if v[index]==0:
            #if assert fails line cannot reach target at given index
            assert(target[index]==a[index])
            return 0
        return (target[index]-a[index])/v[index]

def get_higher_quadratic_solution(a,b,c):
    """
    returns the higher value out of the two solutions of the quadratic equation a*x^2 + b*x + c
    """
    d = b**2-4*a*c
    assert (d>=0) ," discriminant negative, no real solution to this quadratic equation"
    return (-b +np.sqrt(d))/(2*a)

def get_radius_intersection(v,a,r):
    """
    given v,a, such that v*t+a forms a line and a radius r return t where the intersection lies
    """
    #create quadratic from r^2 = norm(v*t+a)^2 for t
    return get_higher_quadratic_solution(v[0]**2+v[1]**2, 2*(a[0]*v[0]+a[1]*v[1]), a[0]**2 + a[1]**2 - r**2)

def get_radius_cut_intersections(r,x,y) -> tuple[float,float]:
    """
    given a radius,x cutoff and y cutoff returns point on circle of radius r such that
    either given x or given y matches, the point with smaller x value is returned
    """
    assert (x>=0 and y>=0), "cutoffs need to be positive"
    point1= (x, get_circle_intersect_x(r,x)) if r>x else (r,0)
    point2= (get_circle_intersect_x(r,y),y) if r>y else (r,0)
    return point1 if x<point2[0] else point2
def get_circle_path_stop(r,x,y,alpha,offset) -> tuple[float,float]:
    """
    path starting from (0,r) clockwise along the circle of radius r returns first intersection between
    either x_cutoff, y_cutoff or the line formed by y=offset + x*0 rotated by alpha
    """
    point1 = get_radius_cut_intersections(r,x,y)
    rot_mat = Rot.from_euler('z', alpha, degrees=True)  # z-axis for 2D rotation
    rot_mat = rot_mat.as_matrix()[:2, :2]
    point2 = rot_mat  @ np.array((get_circle_intersect_x(r,offset),offset))
    point2 = tuple(point2)
    return point1 if point1[0]<point2[0] else point2

def get_cirle_point(r,angle):
    ang_rad = np.radians(angle)
    return (r*np.cos(ang_rad),r*np.sin(ang_rad))

class DiskWithStopShape(Shape):
    """
    defines a diskshape with inner radius, outer radius and 2 points in first quadrant on that radii which acts as a stop 
    by design this shape is symmetric to the x axis, full disk from (-rOut / -rIn,0) to (0,rOut /rINn) then going till it hits the line defined by the two points

    """
    def __init__(self,color,label, rIn,rOut, stoppers:tuple[tuple[float,float],tuple[float,float]]):
        self.color = color
        self.label = label
        self.rIn,self.rOut = np.sort((rIn,rOut))
        assert stoppers[0] != stoppers[1] , "the two points need to be different"
        assert np.allclose((self.rIn,self.rOut),tuple(np.sort([np.linalg.norm(p) for p in stoppers]))) , "the stoppers need to be on the two radii"
        assert np.all([p[1]>0 for p in stoppers]) , "stopper points need to be in the upper half plane"
        self.stoppers = stoppers

    def get_line_parameters(self):
        a,b = self.stoppers
        a = np.array(a)
        b = np.array(b)
        return b-a,a

    def point_is_in(self,point):
        x,y  = point
        r_p = np.linalg.norm(point)
        if self.rIn<=r_p and r_p <= self.rOut:
            v,a = self.get_line_parameters()
            t = get_radius_intersection(v,a,r_p)
            stopperpoint = v*t + a
            angle_rad = np.arctan2(np.abs(y), x)
            stopper_angle = np.arctan2(stopperpoint[1],stopperpoint[0])
            return angle_rad >= stopper_angle
        return False
    

class CaloShape4PointDirect(Shape):
    """
    2d shape mirrored across x axis. General form is a ring with rIn and rOut radius start/stop , in the first quadrant stopped by
    a x_cutoff,y_cutoff and a line (referenced as cone line) formed via a reference angle and an offset (upwards) 
    """
    def __init__(self, color,label,rIn,rOut,x_cutoff,y_cutoff,angle,cone_offset,show_mark=False):
        self.color = color
        self.label = label
        self.angle = angle
        self.x_cutoff = x_cutoff
        self.y_cutoff = y_cutoff
        self.cone_offset = cone_offset
        self.rIn = rIn
        self.rOut = rOut
        self.show_mark = show_mark

    def should_points_be_marked(self):
        return self.show_mark
    
    def mark_defining_points(self,ax):
        r1,r2 = self.get_radius_points()
        c1,c2 = self.get_cone_points()

        pointList = [
            (r1,f"inner rad: {self.label}","cyan"),
            (r2,f"outer rad: {self.label}","orange"),
            (c1,f"inner cone: {self.label}","brown"),
            (c2,f"outer cone: {self.label}","black")
        ]
        for p,label,color in pointList:
            ax.plot(*p, 'o', color=color,label = label)

    def get_rotation(self,angle = None):
        angle = angle if angle else self.angle
        r = Rot.from_euler('z', angle, degrees=True)  # z-axis for 2D rotation
        return r.as_matrix()[:2, :2]
    
    def get_cone_direction_offset(self):
        """
        returns vectors v,a for the cone line such that v*t + a is the line of the cone parameterized by t
        """
        r = self.get_rotation()
        v = r @ np.array((1,0))
        a = r @ np.array((0,1))*self.cone_offset
        return v,a
    
    def get_radius_points(self):
        """
        returns two points , one on the inner circle, other on the outer circle

        resulting points are found via fucntion get_circle_path_stop
        """
        p_in = get_circle_path_stop(self.rIn,self.x_cutoff,self.y_cutoff,self.angle,self.cone_offset)
        p_out = get_circle_path_stop(self.rOut,self.x_cutoff,self.y_cutoff,self.angle,self.cone_offset)
        return p_in,p_out
        
    
    def get_cone_points(self) -> tuple[tuple[float,float],tuple[float,float]]:
        v,a = self.get_cone_direction_offset()
        cut_point =(self.x_cutoff,self.y_cutoff)
        t_x = get_intersection(v,a,cut_point,0)
        t_y = get_intersection(v,a,cut_point,1)

        t_rIn = get_radius_intersection(v,a,self.rIn)
        t_rOut = get_radius_intersection(v,a,self.rOut)

        t_in = max(t_y,t_rIn)
        t_out = min(t_x,t_rOut)

        #smooth and exact transition between cases
        if t_in<t_out:
            return tuple(v*t_in+a),tuple(v*t_out+a)
        else:
            #in this case the cutoff point lies above the cone
            return cut_point,cut_point

    def point_is_in(self,point):
        r_p = np.linalg.norm(point)
        rad_check = self.rIn<=r_p and r_p <= self.rOut
        point = (point[0],np.abs(point[1]))
        x,y = point
        cutoff_checks = y>=self.y_cutoff and x<=self.x_cutoff
        cone_check = (self.get_rotation(-self.angle) @ point)[1]>=self.cone_offset
        return rad_check and (x<0 or (cutoff_checks and cone_check))
    

class CaloShape4Point(CaloShape4PointDirect):
    """
    2d shape mirrored across x axis. General form is a ring with rIn and rOut radius start/stop , in the first quadrant stopped by
    a x_cutoff,y_cutoff and a line (referenced as coneline) formed via a reference angle and an offset (upwards) 

    instead of giving the y_cutoff, calculate it via the wished for flat_length, the length of the line paralell to the x-axis, 
    from the inner radius to the coneline
    """
    def __init__(self, color,label,rIn,rOut,x_cutoff,flat_length,angle,cone_offset,show_mark=False):
        self.color = color
        self.label = label
        self.angle = angle
        self.x_cutoff = x_cutoff
        self.cone_offset = cone_offset
        self.rIn = rIn
        self.rOut = rOut
        self.show_mark = show_mark

        self.y_cutoff = self.calc_ycutoff_from_flat_length(flat_length)
    
    def calc_ycutoff_from_flat_length(self,flat_length):
        """
        given the flat_length and already defined radii , x cutoff refangle and cone offset calculate the needed y cutoff, throws error if not possible
        """
        rad_angle = np.radians(self.angle)
        # starting from points R (parameterised by rIn, angle beta), at an initial refangle 0 with offset m = cone_offset, targetpoint M with y(M)=m
        # with line R + v = M given v = (cos(angle),-sin(angle))*flat_length , solve for beta
        sin_inner_radius_point_angle = self.cone_offset/self.rIn + flat_length/self.rIn *np.sin(rad_angle)
        flat_length_too_big= f"flat_length ({flat_length:2.1f}) too big to find an y_cutoff given inner radius {self.rIn:2.3f} , cone offset {self.cone_offset:2.3f} and reference angle {self.angle:2.3f}"
        assert (sin_inner_radius_point_angle <= 1) , flat_length_too_big
        beta = np.arcsin(sin_inner_radius_point_angle)
        assert (np.pi/2 - rad_angle>beta) , "found unwished solution (inner radius point would have negative x) : " + flat_length_too_big
        # rotate and get height
        return np.sin(rad_angle +beta ) * self.rIn


class CaloShape(CaloShapeDirect):
    
    def __init__(self, color,label,rIn,rOut,ref_angle,inner_slope_offset,outer_slope_offset, x_cutoff):
        self.color = color
        self.label = label
        self.rIn = rIn
        self.rOut = rOut
        self.ref_angle = ref_angle
        self.inner_slope_offset = inner_slope_offset
        self.outer_slope_offset = outer_slope_offset
        self.x_cutoff = x_cutoff
        self.innerpoint,self.outer_slope_point,self.outerpoint = self.get_3mainpoints_for_caloshape(rIn,rOut,ref_angle,inner_slope_offset,outer_slope_offset, x_cutoff)

    def get_3mainpoints_for_caloshape(self,rIn,rOut,ref_angle,inner_slope_offset,outer_slope_offset, x_cutoff):
        innerpoint = np.array((get_circle_intersect_x(rIn,inner_slope_offset),inner_slope_offset))
        outerpoint = np.array((get_circle_intersect_x(rOut,outer_slope_offset),outer_slope_offset))

        r = Rot.from_euler('z', ref_angle, degrees=True)  # z-axis for 2D rotation
        rot_matrix = r.as_matrix()[:2, :2]
        innerpoint = rot_matrix @ innerpoint
        outer_slope_point = rot_matrix @ outerpoint

        if outer_slope_point[0]<x_cutoff:
            outerpoint=outer_slope_point
        else:
            outerpoint=np.array((x_cutoff,get_circle_intersect_x(rOut,x_cutoff)))
            m,a = line_from_points(innerpoint,outer_slope_point)
            outer_slope_point = np.array((x_cutoff,m*x_cutoff+a))
        return innerpoint,outer_slope_point,outerpoint
    
    
    
    def plot_outline(self):
        
        plt.figure(figsize=(6, 6))
        rIn,rOut,inner,out_slope,outer = self.rIn,self.rOut,self.innerpoint,self.outer_slope_point,self.outerpoint

        plt.plot(*plot_halfcircle_from_point(rIn,inner), label='inner radius')
        plt.plot(*plot_halfcircle_from_point(rOut,outer), label='outer radius')
        if out_slope[0]==outer[0]:
            plt.vlines(out_slope[0], out_slope[1], outer[1], label='x_cutoff line')
        plt.plot(*zip(inner,out_slope), label='slope line')

        r = Rot.from_euler('z', self.angle, degrees=True)  # z-axis for 2D rotation
        rot_matrix = r.as_matrix()[:2, :2]
        ref_angle_point = 1.2*rOut* rot_matrix @ np.array((1,0))
        plt.plot(*zip((0,0),ref_angle_point),linestyle='dotted', label='angle ref line')

        # Equal aspect ratio so the circle isn't distorted
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.legend()
        plt.title('mockup general shape of calorimeter volumes')
        plt.show()

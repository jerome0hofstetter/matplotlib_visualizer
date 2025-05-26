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

def last_point_redundant(res_edges,edge, tol=1e-8):
    """
    checks if the last added point is redundant or not, if that point is on the same line as the line formed by secondlast point and new point

    throws an assertion error if new point is on the same line as the two points but between the last two points

    assumes res_edges contains unique points
    """
    if len(res_edges)<2:
        return False
    A = np.array(res_edges[-2])
    B = np.array(res_edges[-1])
    P = np.array(edge)
    
    
    AB = B - A
    AP = P - A
    
    cross = AB[0]*AP[1] - AB[1]*AP[0]  # scalar cross product in 2D
    cross = cross / np.linalg.norm(AB) / np.linalg.norm(AP) # norm crossproduct
    if np.abs(cross)<tol:
        assert np.linalg.norm(AB) <=np.linalg.norm(AP), "the new point is on the line formed by the last two points AND between those two points"
        return True
    return False
    
def cleanup_polygon_edges(edges):
    """
    returns a new list of edges with all the unnecessary edges removed: same edge back to back are removed and
    if 3 edges are on the same line the middle edge is not needed
    """
    res_edges = []
    for edge in edges:
        if len(res_edges)==0:
            res_edges.append(edge)
        elif not np.allclose(res_edges[-1], edge, rtol=1e-9, atol=1e-12):
            if last_point_redundant(res_edges,edge):
                res_edges.pop()
            res_edges.append(edge)
    if last_point_redundant(res_edges,res_edges[0]):
        res_edges.pop(0)
    return res_edges


def get_simple_polygon_intersection(edges,edge):
    """
    given a list of edges forming a simple polygon, no self intersections, max 2 points which have the same x value and any vertical line intersects 1 (at an other edge) or two times

    tries to find the intersection with another
    """
    x,y = edge
    n = len(edges)
    for i in range(n):
        (x1, y1) = edges[i]
        (x2, y2) = edges[(i + 1) % n]  # wrap around

        if (x1<x and x<x2) or (x1> x and x > x2) :
            t = (x - x1) / (x2 - x1)
            y_intersection = y1 + t * (y2 - y1)
            return (min(y_intersection,y), max(y_intersection,y))

    return None

def pointline_intersect(p ,q):
    """
    given 2 sets of points, each forming a line segment from one to the other
    return if they intersect or not


    """
    p = [np.array(s) for s in p]
    q = [np.array(s) for s in q]
    v_1,a_1 = p[1] - p[0],p[0]
    v_2,a_2 = q[1] - q[0],q[0]
    # v_1 t_1 + a_1 = v_2 t_2 + a_2  -> a_1 - a_2 = v_2 t_2 - v_1 t_1 = V @ t
    V = np.column_stack((-v_1,v_2))
    A = a_1 - a_2
    det = np.linalg.det(V)
    if np.abs(det) > 1e-10:
        t = np.linalg.solve(V, A)
        if 0<=t[0]<=1 and 0<=t[1]<=1:
            print(f"found intersection for : {p} with {q} at {a_1 + v_1 * t[0]} with t : {t}")
            return True
    return False

def has_self_intersections(edges):
    n = len(edges)
    def get_edge_pair(r):
        return (edges[r % n],edges[(r+1) % n])
    for i in range(n):
        for j in range(n-3):
            if pointline_intersect(get_edge_pair(i),get_edge_pair(i+j+2)):
                return True
    return False

def is_sectionable_polygon(edges,allow_intersections = False):
    n = len(edges)
    if n<3:
        return False
    start_index = np.argmin([p[0]] for p in edges)
    if edges[(start_index+1)% n ][0] == edges[start_index][0]:
        start_index +=1
    dirr  =1
    # starts at point with smallest x, iterates over, looks for direction changes, allows for 1 directionchange, 
    # a second one right at the end (for edgecase where there are two edges with the same minimal x)
    for i in range(n):
        point1 = edges[(start_index+i)% n ]
        point2 =  edges[(start_index+1+i)% n ]
        if dirr* point1[0]>=point2[0] * dirr:
            dirr *= -2
            if abs(dirr)>3 and i<n-1:
                return False

    return not has_self_intersections(edges) or allow_intersections

def get_sectioning_of_simple_polygon(edges):
    sections = []
    lookup_dict = dict()
    edges = cleanup_polygon_edges(edges)
    if not is_sectionable_polygon(edges):
        print("polygon is not simple or not sectionable")
    #assert is_simple_sectionable_polygon(edges), "polygon is not simple or not sectionable"

    edges_sorted = sorted(edges,key = lambda x:x[0])
    # first try to find sections consisting of two points at the same x axis, just different y axis
    # error if for same x axis more than 2 points are found

    for edge in edges:
        x,y = edge
        if x in lookup_dict:
            assert len(lookup_dict[x])==1 , " max 2 points that have the same x value"
            lookup_dict[x].append(y)
            lookup_dict[x].sort()
            sections.append((x,lookup_dict[x]))

        else:
            lookup_dict[x] = [y]

        # find intersection for that edge to get a section corresponding to the edge, misses any edges themself
        y_pair = get_simple_polygon_intersection(edges,edge)
        if y_pair:
            assert len(lookup_dict[x])==1, "not supported case, 2 edges with same x and polygon has an intersection with vertical line at same x that isnt an edge"
            sections.append((edge[0],y_pair))

    # sort sections
    sections.sort(key=lambda x: x[0])
    start_edge = edges_sorted[0]
    end_edge = edges_sorted[-1]

    #add in the start and end edge, unless foreach start and/or end has 2 points with same x
    if sections[0][0]!= start_edge[0]:
        sections.insert(0,(start_edge[0],(start_edge[1],start_edge[1])))
    if sections[-1][0]!= end_edge[0]:
        sections.append((end_edge[0],(end_edge[1],end_edge[1])))
    return sections




class DiskWithStopShape(Shape):
    """
    defines a diskshape with inner radius, outer radius and 2 points in first quadrant on that radii which acts as a stop 
    by design this shape is symmetric to the x axis, full disk from (-rOut / -rIn,0) to (0,rOut /rINn) then going till it hits the line defined by the two points

    """
    def __init__(self,color,label, rIn,rOut, stoppers:tuple[tuple[float,float],tuple[float,float]]):
        self.color = color
        self.label = label
        self.rIn,self.rOut = np.sort((rIn,rOut))
        assert np.allclose((self.rIn,self.rOut),tuple([np.linalg.norm(p) for p in stoppers])) , "the stoppers need to be on the two radii"
        assert np.all([p[1]>0 for p in stoppers]) , "stopper points need to be in the upper half plane"
        self.stoppers = stoppers
        #with this first stopper is in the inner radius, second on the outer radius, else the second assert will throw an error

        #
        v,a = self.get_line_parameters()
        line_angle = np.arctan2(v[1],v[0])
        innerpoint = self.stoppers[0]
        innerpoint_angle = np.arctan2(innerpoint[1],innerpoint[0])
        angle_diff = line_angle-innerpoint_angle
        # looking at the tangentline of inner radius at the inner stopper, rotate to have that point on x axis, limit the angle then between -180 to 180 degrees
        assert -np.pi/2<=angle_diff and angle_diff <= np.pi/2, "the stoppers are placed such that the line formed by those two intersects the inner radius at a 3. point between the stoppers"

    def get_line_parameters(self):
        a,b = self.stoppers
        a = np.array(a)
        b = np.array(b)
        return b-a,a
    
    def get_cutout_shape_pcone_params(self):
        """
        returns if necessary the parameters of a cutout shape (polygon) that are needed to create this shape as the result of 
        subtracting away the cutout shape from a optimised makesphere shape

        a makesphere shape is a shape generated from parameterising in spherical coordinates radius r in [rIn,rOut], 
        polar angle theta in [0,180-theta_cut] (0 is at negative x-axis) and azimuthal angle phi in [0,360] (full rotation)

        optimised for theta_cut such that a minimum needs to be cutout
        """
        inner,outer = self.stoppers
        inner = np.array(inner)
        outer = np.array(outer)
        assert np.all([p[0]>=0 for p in self.stoppers]), "stoppers are assumed to have nonnegative x value"
        inner_angle,outer_angle = np.arctan2(inner[1],inner[0]), np.arctan2(outer[1],outer[0])

        theta_cut = min(inner_angle,outer_angle)
        v,a = outer-inner,inner
        points =  [v*t+a for t in [-0.01,1.01]]
        if inner_angle> outer_angle:
            lowerlimit = np.sin(theta_cut)*self.rIn- np.linalg.norm(v)*0.01
            return [(x,(lowerlimit,y)) for (x,y) in points]
        elif inner_angle<outer_angle :
            upperxlimit = np.cos(theta_cut)*self.rOut +np.linalg.norm(v)*0.01
            edges = [points[0], points[1], (upperxlimit,points[1][1]), (upperxlimit,points[0][1]) ]
            return get_sectioning_of_simple_polygon(edges)
        return None
    
    def get_cutout_shapes(self):
        """
        calculates the cutoutshape and also the ghost shape and returns it
        """
        pcone_param = self.get_cutout_shape_pcone_params()
        if pcone_param:
            edges = pcone_param_to_polygon(self.get_cutout_shape_pcone_params())
            cutout = PolygonShape("red","cutout",edges)

            inner,outer = self.stoppers
            inner_angle,outer_angle = np.arctan2(inner[1],inner[0]), np.arctan2(outer[1],outer[0])
            ref_angle = min(inner_angle,outer_angle)
            ghost = CaloShape4PointDirect((0.082, 0.929, 0.8, 0.38),"ghost",self.rIn,self.rOut,self.rOut*2,0,np.degrees(ref_angle),0)


            return [cutout,ghost]
        return []


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


def sphere_cutout_volume(r,a,b):
    """
    given a sphere of radius r and a start-/endcutoff a / b lets say x axis 

    return the volume of the cutout
    """
    assert a<=b, "b>=a is enforced"
    assert r>=0, "positive radius"
    a= max(a,-r)
    b= min(b,r)

    return np.pi*(r**2 *(b-a) - 1/3 * (b**3-a**3))

def pcone_param_to_polygon(pcone: list[tuple[float,tuple[float,float]]]):
    """
    pcone parameter are a list of tuples of the form (x,y_min,y_max), this returns based on that a list of points that are the edegs of the polygon
    """
    forward = [(x,y_min) for x,(y_min,y_max) in pcone]
    backward = [(x,y_max) for x,(y_min,y_max) in pcone]
    backward.reverse()
    forward.extend(backward)
    return forward


def fullcone_volume(m,k,a,b):
    """
    seeing the cone as a rotationvolume of the area under a line from start-/endcut a / b

    given said lines m (slope) and k (offset) , such that m*x+k forms said line and a / b (start/end)    
    
    return the volume of this cone
    """
    return np.pi * 1/3 * m**2 * (b**3 - a**3) + m * k * ( b**2 - a**2) + k**2 * ( b - a)

def sphere_volume(r):
    return 4/3 * np.pi * r**3

def calo4point_rotation_volume(rInner,rOuter,innerRadiusPoint, innerConePoint,outerConePoint):
    inner_sliver = sphere_cutout_volume(rInner,innerRadiusPoint[0],rInner)
    outer_sliver = sphere_cutout_volume(rOuter,outerConePoint[0],rOuter)
    hollow_sphere = sphere_volume(rOuter) - sphere_volume(rInner)
    cone_volume_cutout = 0
    if innerConePoint!=outerConePoint:
        m,a = line_from_points(innerConePoint,outerConePoint)
        cone_volume_cutout = fullcone_volume(m,a,innerConePoint[0],outerConePoint[0])


    ycutoff_volume = 0
    if innerRadiusPoint != innerConePoint:
        m,a = line_from_points(innerRadiusPoint,innerConePoint)
        assert abs(m)<1e-14, f"should be by design , {m}"
        ycutoff_volume = fullcone_volume(0,innerRadiusPoint[1], innerRadiusPoint[0],innerConePoint[0] )
    return hollow_sphere + inner_sliver - outer_sliver - cone_volume_cutout - ycutoff_volume


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
    

    def cutout_cones_parameter(self):
        pIn,pOut = self.get_radius_points()
        cIn,cOut = self.get_cone_points()
        ref_angle = np.arctan2(cOut[1],cOut[0])
        shapes = []
        if cIn[0]!= pIn[0]:
            shapes.append([(0.99*pIn[0],(0,pIn[1])) ,( 1.01*cIn[0],(0,cIn[1])) ])
        if cIn[0] != cOut[0]:
            cIn,cOut = np.array(cIn), np.array(cOut)
            v = cOut - cIn
            points = [ v*t + cIn for t in [-0.01,1.01]]
            shapes.append([(x,(0,y)) for x,y in points])
        if cOut[1]!= pOut[1]:
            shapes.append([(pOut[0],(0,pOut[1]*1.01)) ,( 1.01*self.rOut*np.cos(ref_angle)*1.01,(0,pOut[1]*1.01)) ])
        return shapes

    def get_cutouts(self):
        cone_params = self.cutout_cones_parameter()

        shapes = []
        for cone_param in cone_params:
            edges = pcone_param_to_polygon(cone_param)
            color = (0.929, 0.082, 0.082, 0.561)
            shape = MirrorPolygonShape(color,"cutout",edges)
            shapes.append(shape)
        return shapes



    
    def get_rotation_volume(self):
        return calo4point_rotation_volume(self.rIn,self.rOut,self.get_radius_points()[0] ,  *self.get_cone_points())
    
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

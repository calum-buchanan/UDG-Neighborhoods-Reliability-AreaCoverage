'''
This file contains functions related to finding regions of intersection of circles or tori in the plane (equivalently, finding neighborhoods for a vertex added to a unit disk graph with an optional buffer distance around the vertex).
It also contains functions with which one can obtain relative estimates of reliability and all-terminal reliability, as well as functions which use the former algorithms to determine the location to add or move a vertex in a unit disk graph to maximize the reliability of the resulting graph.
See the papers "Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms" and "Node Placement to Maximize Reliability of a Communication Network with Application to Satellite Swarms" by Calum Buchanan, Puck Rombach, James Bagrow, and Hamid R. Ossareh.
'''

import math
from dataclasses import dataclass
from typing import Optional
from collections import namedtuple
from itertools import combinations
import numpy as np
from numpy.linalg import matrix_rank
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import matplotlib.pyplot as plt


rng = np.random.default_rng()


'''
FINDING NEIGHBORHOODS FOR A VERTEX ADDED TO A UNIT DISK GRAPH
'''

@dataclass
class Point:
  x: float
  y: float
  label: int
  z: Optional[float] = None

  @property
  def is_2d(self):
    return self.z is None
  @property
  def is_3d(self):
    return self.z is not None


@dataclass
class CyclicInterval:
  min: Optional[float]
  max: Optional[float]
  label: Point
  disappear: Optional[float] = None
  reappear: Optional[float] = None


@dataclass
class Endpoint:
  coord: float
  point: float
  mtype: str


def points(G):
  """
  Input: nx.random_geometric_graph G
  Output: list of locations of the vertices of G in the form of the data class "Point".
  """
  centers = []
  for i in range(len(G.nodes)):
    centers.append(Point(G.nodes[i]['pos'][0], G.nodes[i]['pos'][1], i))
  return centers


def perturb_points(points: list, max_noise: float) -> list:
  """
  Adds a small random perturbation to the coordinates of each point in the list.
  """
  for p in points:
    p.x = p.x + np.random.uniform(-max_noise, max_noise)
    p.y = p.y + np.random.uniform(-max_noise, max_noise)
    if p.is_3d:
      p.z = p.z + np.random.uniform(-max_noise, max_noise)
  return points



"""
Find regions defined by circles in the plane
"""


def sets_from_intervals(iv_list: list) -> list:
  """
  Build a list of sets describing interval overlap, the first element in each set is an endpoint of an interval.
  """
  # We list the endpoints and their corresponding interval indices
  endpts = (
      [Endpoint(interval.min, interval.label, 'min') for interval in iv_list] + 
      [Endpoint(interval.max, interval.label, 'max') for interval in iv_list]
  )
  # We sort the endpoints from left to right
  endpts_sorted = sorted(endpts, key=lambda Endpoint : Endpoint.coord)
  sets = []
  current_set = [interval.label for interval in iv_list if interval.max < interval.min]
  # We run through the endpoints from left to right
  for endpt in endpts_sorted:
    # We hit the end of a current interval; remove it from current
    if endpt.mtype=='max':
      current_set.remove(endpt.point)
      sets = sets + [[endpt.point] + current_set.copy()]
    # We hit a new interval; add it to current
    else:
      current_set = [endpt.point] + current_set
      sets = sets + [current_set.copy()]
  if sets == []:
    sets = [[]]
  return sets


def interval_from_sweep(center: Point, point: Point, radius: float) -> CyclicInterval:
  """
  Sweep a circle of a given radius around the origin and record the entry-exit interval of point.
  """
  dx = point.x - center.x
  dy = point.y - center.y
  a = math.atan(dy / abs(dx))
  b = math.acos(math.sqrt(dx**2 + dy**2) / (2*radius))
  if dx < 0:
    IN = math.pi - (a + b)
    if IN > math.pi:
      IN = IN - 2*math.pi
    OUT = math.pi - (a - b)
    if OUT > math.pi:
      OUT = OUT - 2*math.pi
    return CyclicInterval(min=IN, max=OUT, label=point)
  return CyclicInterval(min=a - b, max=a + b, label=point)


def find_regions(centers: list, radius: float, maximal_only=False) -> list:
  """
  Finds the set of feasible regions of overlap given a set of centers and circles of radius around them.
  """
  centersp = perturb_points(centers, 0.01)
  # centersp = centers
  R = [[]]
  for i in centersp:
    Li = []
    for j in [center for center in centersp if center != i]:
      if math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2) <= 2*radius:
        Li.append(interval_from_sweep(i, j, radius))
    for s in sets_from_intervals(Li):
      R.append(s)
      R.append(s + [i])
      R.append(s[1:])
      R.append(s[1:] + [i])
  res = []
  for r in R:
    r_sorted = sorted(r, key=lambda point : point.x)
    if r_sorted not in res:
      res.append(r_sorted)
  if maximal_only == True:
    maximalres = []
    reslabels = [[p.label for p in region] for region in res]
    l = len(reslabels)
    for i in range(l):
      if not any([set(reslabels[i]) < set(reslabels[j]) for j in range(l)]):
        maximalres.append(res[i])
    return maximalres
  return res



"""
Find regions defined by 2D tori in the plane
(possble neighborboods for a vertex added with a buffer)
"""


def sets_from_buffer_intervals(iv_list: list) -> list:
  """
  Build a list of sets describing interval overlap, the first element in each set is an endpoint of an interval.
  """
  # We list the endpoints and their corresponding interval indices
  endpts = (
      [Endpoint(interval.min, interval.label, 'in') for interval in iv_list if interval.min is not None] + 
      [Endpoint(interval.max, interval.label, 'out') for interval in iv_list if interval.max is not None] + 
      [Endpoint(interval.reappear, interval.label, 'reappear') for interval in iv_list if interval.reappear is not None] +
      [Endpoint(interval.disappear, interval.label, 'disappear') for interval in iv_list if interval.disappear is not None]
  )
  # We sort the endpoints from left to right
  endpts_sorted = sorted(endpts, key=lambda Endpoint : Endpoint.coord)
  sets = []
  # current_set is for points in annulus with nothing in its inner circle
  current_set = []
  # running_set is for points in annulus that might have others in its inner circle
  running_set = []
  for interval in iv_list:
    if interval.reappear is None:
      if interval.max < interval.min:
        running_set.append(Endpoint(interval.min, interval.label, 'in'))
    else:
      if interval.max is None and not interval.reappear < interval.disappear:
        running_set.append(Endpoint(interval.reappear, interval.label, 'reappear'))
      if interval.max is not None and interval.max < interval.min and not interval.reappear < interval.disappear:
        if interval.disappear > 0:
          running_set.append(Endpoint(interval.reappear, interval.label, 'reappear'))
        else:
          running_set.append(Endpoint(interval.min, interval.label, 'in'))
  # create a 'bad' list of points in the inner circle of the annulus
  bad = []
  for interval in iv_list:
    if interval.reappear is not None and interval.reappear < interval.disappear:
        bad.append(Endpoint(interval.disappear, interval.label, 'disappear'))
  if len(bad)==0:
    current_set = running_set
  # We run through the endpoints from left to right
  for endpt in endpts_sorted:
    # A point leaves; remove it from current_set
    if endpt.mtype=='out':
      running_set = [c for c in running_set if c.point is not endpt.point]
      if len(bad) == 0:
        current_set = running_set
        sets = sets + [[endpt] + current_set.copy()]
    # A point enters; add it to current_set
    if endpt.mtype == 'in':
      running_set = [endpt] + running_set
      if len(bad) == 0:
        current_set = running_set
        sets = sets + [current_set.copy()]
    # A bad point becomes good; add it to running_set
    if endpt.mtype == 'reappear':
      bad = [c for c in bad if c.point is not endpt.point]
      running_set = [endpt] + running_set
      if len(bad) == 0:
        current_set = running_set
        sets = sets + [current_set.copy()]
    # A good point becomes bad; remove it from running_set
    if endpt.mtype == 'disappear':
      running_set = [c for c in running_set if c.point is not endpt.point]
      if len(bad) == 0:
        current_set = running_set
        sets = sets + [[endpt] + current_set.copy()]
      bad = [endpt] + bad
  if sets == []:
    sets = [[]]
  return sets


def interval_from_outer_sweep(center: Point, point: Point, radius: float, buffer: float) -> CyclicInterval:
  '''
  Sweep a (buffer, radius)-torus around a point 'center' fixed on its OUTER boundary. Return a CyclicInterval of angles at which a point 'point' enters and exits the outer and inner boundaries of the torus during the sweep
  '''
  dx = point.x - center.x
  dy = point.y - center.y
  a1 = math.atan(dy / abs(dx))
  b1 = math.acos(math.sqrt(dx**2 + dy**2) / (2*radius))
  if dx < 0:
    IN = math.pi - (a1 + b1)
    if IN > math.pi:
      IN = IN - 2*math.pi
    OUT = math.pi - (a1 - b1)
    if OUT > math.pi:
      OUT = OUT - 2*math.pi
  if dx >= 0:
    IN = a1 - b1
    OUT = a1 + b1
  ro = radius
  ri = buffer
  t = (dx**2 + dy**2 - (ro - ri)**2) / (2*ro)
  a = math.sqrt(ri**2 - (t - ri)**2)
  b = t + ro - ri
  r = math.sqrt(a**2 + b**2)
  alpha = math.atan(b/-a) + math.pi
  d1 = math.acos(dy/r) + math.atan(b/a)
  d2 = math.atan(b/a) - math.acos(dy/r)
  e1 = math.acos(dy/r) + alpha
  e2 = alpha - math.acos(dy/r)
  if dx <= 0:
    dis = d1
    re = e1
  else:
    dis = d2
    re = e2
  if dis > math.pi:
    dis = dis - 2*math.pi
  if re > math.pi:
    re = re - 2*math.pi
  return CyclicInterval(min=IN, max=OUT, label=point, disappear = dis, reappear = re)


def interval_from_inner_sweep(center: Point, point: Point, radius: float, buffer: float) -> CyclicInterval:
  '''
  Sweep a (buffer, radius)-torus around a point 'center' fixed on its INNER boundary. Return a CyclicInterval of angles at which a point 'point' enters and exits the outer and inner boundaries of the torus during the sweep
  '''
  dx = point.x - center.x
  dy = point.y - center.y
  a1 = math.atan(dy / abs(dx))
  b1 = math.acos(math.sqrt(dx**2 + dy**2) / (2*buffer))
  if dx < 0:
    dis = math.pi - (a1 + b1)
    if dis > math.pi:
      dis = dis - 2*math.pi
    re = math.pi - (a1 - b1)
    if re > math.pi:
      re = re - 2*math.pi
  if dx >= 0:
    dis = a1 - b1
    re = a1 + b1
  ro = radius
  ri = buffer
  t = (dx**2 + dy**2 - ro**2 + ri**2) / (2*ri)
  a = math.sqrt(ro**2 - (t - ri)**2)
  b = t
  r = math.sqrt(a**2 + b**2)
  alpha = math.atan(b/-a) + math.pi
  d1 = math.acos(dy/r) + math.atan(b/a)
  d2 = math.atan(b/a) - math.acos(dy/r)
  e1 = math.acos(dy/r) + alpha
  e2 = alpha - math.acos(dy/r)
  if dx <= 0:
    IN = d1
    OUT = e1
  else:
    IN = d2
    OUT = e2
  if IN > math.pi:
    IN = IN - 2*math.pi
  if OUT > math.pi:
    OUT = OUT - 2*math.pi
  return CyclicInterval(min=IN, max=OUT, label=point, disappear = dis, reappear = re)


def find_regions_outer_sweeps(centers: list, radius: float, buffer: float, maximal_only=False) -> list:
  """
  Finds the set of feasible regions of overlap with buffer given a set of centers and circles of radius around them.
  """
  centersp = perturb_points(centers, 0.001)
  # centersp = centers
  # R = [([],'infinity')]
  R = []
  for i in centersp:
    Li = []
    for j in [center for center in centersp if center != i]:
      if radius - buffer >= math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2):
        Li.append(interval_from_sweep(i, j, radius))
      if radius + buffer <= math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2) < 2*radius:
        Li.append(interval_from_sweep(i, j, radius))
      if radius - buffer < math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2) < radius + buffer:
        Li.append(interval_from_outer_sweep(i, j, radius, buffer))
    S = sets_from_buffer_intervals(Li)
    if not any(S):
      R.append(([i], (None, i, 'outer sweep')))
    else:
      for s in S:
        if len(s) > 0:
          R.append(([j.point for j in s], (s[0].coord, i, 'outer sweep')))
          R.append(([j.point for j in s] + [i], (s[0].coord, i, 'outer sweep')))
          # Points on the boundary of inner circle must be included in the region
          # Points on the boundary of outer circle may or may not be included
          if s[0].mtype not in ['disappear', 'reappear']:
            R.append(([j.point for j in s[1:]], (s[0].coord, i, 'outer sweep')))
            R.append(([j.point for j in s[1:]] + [i], (s[0].coord, i, 'outer sweep')))
  res = []
  for r in R:
    r_sorted = sorted(r[0], key=lambda point : point.x)
    res.append((r_sorted, r[1]))
  if maximal_only == True:
    maximalres = []
    reslabels = [[p.label for p in region[0]] for region in res]
    maximalreslabels = []
    l = len(reslabels)
    for i in range(l):
      Ri = reslabels[i]
      if not any([set(Ri) < set(reslabels[j]) for j in range(l)]):
        if Ri not in maximalreslabels:
          maximalres.append(res[i])
          maximalreslabels.append(Ri)
    return maximalres
  return res


def find_regions_inner_sweeps(centers: list, radius: float, buffer: float, maximal_only=False) -> list:
  """
  Finds the set of feasible regions of overlap with buffer given a set of centers and circles of radius around them.
  """
  centersp = perturb_points(centers, 0.001)
  # centersp = centers
  # R = [([], 'infinity')]
  R = []
  for i in centersp:
    Li = []
    for j in [center for center in centersp if center != i]:
      if 2*buffer > math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2) > radius - buffer:
        Li.append(interval_from_inner_sweep(i, j, radius, buffer))
      if 2*buffer <= math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2) < radius + buffer:
        Li.append(interval_from_sweep(i, j, radius))
      if radius - buffer >= math.sqrt((j.x - i.x)**2 + (j.y - i.y)**2):
        I = interval_from_sweep(i, j, buffer)
        J = CyclicInterval(min=None, max=None, label=I.label, reappear=I.max, disappear=I.min)
        Li.append(J)
    for s in sets_from_buffer_intervals(Li):
      if len(s) > 0:
        # i is included, as it's on the boundary of inner circle 
        R.append(([j.point for j in s] + [i], (s[0].coord, i, 'inner sweep')))
        # if the other point on the boundary is on the outer circle, it may or may not be included
        if s[0].mtype not in ['disappear', 'reappear']:
          R.append(([j.point for j in s[1:]] + [i], (s[0].coord, i, 'inner sweep')))
  res = []
  for r in R:
    r_sorted = sorted(r[0], key=lambda point : point.x)
    res.append((r_sorted, r[1]))
  if maximal_only == True:
    maximalres = []
    reslabels = [[p.label for p in region[0]] for region in res]
    maximalreslabels = []
    l = len(reslabels)
    for i in range(l):
      Ri = reslabels[i]
      if not any([set(Ri) < set(reslabels[j]) for j in range(l)]):
        if Ri not in maximalreslabels:
          maximalres.append(res[i])
          maximalreslabels.append(Ri)
    return maximalres
  return res


def find_buffer_regions_max(centers: list, radius: float, buffer: float) -> list:
  '''
  Enumerate all possible MAXIMAL neighborhoods for a vertex added to a unit disk graph G under the constraint that no other vertex is within buffer distance. Also find a location to add a vertex with this neighborhood.
  '''
  L1 = find_regions_outer_sweeps(centers, radius, buffer, maximal_only=False)
  L2 = find_regions_inner_sweeps(centers, radius, buffer, maximal_only=False)
  L = L1 + L2
  #determine if new point is in the outlined region
  L3 = []
  for L_elt in L:
    a = L_elt[1][1].x
    b = L_elt[1][1].y
    if L_elt[1][0] is not None:
      if L_elt[1][2] == 'outer sweep':
        pt = np.array([a + radius*math.cos(L_elt[1][0]), b + radius*math.sin(L_elt[1][0])])
      else:
        pt = np.array([a + buffer*math.cos(L_elt[1][0]), b + buffer*math.sin(L_elt[1][0])])
      if in_hull(pt, np.array([(center.x, center.y) for center in centers])):
        L3.append((L_elt, pt))
  reslabelsL3 = [[p.label for p in region[0][0]] for region in L3]
  maximalL3 = []
  maximalreslabelsL3 = []
  l = len(reslabelsL3)
  for i in range(l):
    Ri = reslabelsL3[i]
    if not any([set(Ri) < set(reslabelsL3[j]) for j in range(l)]):
      if Ri not in maximalreslabelsL3:
        maximalL3.append(L3[i])
        maximalreslabelsL3.append(Ri)
  return maximalL3



'''
APPLICATIONS TO MAXIMIZING RELIABILITY OF A UNIT DISK GRAPH
'''

def ProbGraph(G,Pe,Pv):
    """
    Input:
    -a networkx graph G
    -a dictionary of edge-operation probabilities Pe
    -a dictionary of vertex-operation probabilities Pv
    Output: a subgraph of G induced by the operational vertices and edges
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    n = G.order()
    m = G.size()
    for e in G.edges:
        if rng.binomial(1,Pe[e])==1:
            H.add_edge(e[0],e[1])
    for v in H.nodes:
        if rng.binomial(1,Pv[v])==0:
            H.remove_node(v)
    return H


def ProbGraphATR(G,p):
    """
    Simplified ProbGraph: vertices perfectly reliable, edges operate
    with probability p
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    E = [e for e in G.edges]
    m = len(E)
    OperationalEdges = rng.binomial(1,p,m)
    for i in range(m):
      if OperationalEdges[i]==1:
        H.add_edge(E[i][0], E[i][1])
    return H


def MonteRel(G,Pe,Pv,epsilon,delta):
    """
    Input:
    -a graph G
    -a dictionary of edge-operation probabilities Pe
    -a dictionary of vertex-operation probabilities Pv
    -epsilon and delta > 0 and epsilon ≤ Rel(G)
    Output: with probability at least 1-delta, a number within 
    epsilon*(1-Rel(G)) of the probability ProbGraph(G,Pe,Pv) is connected.
    """
    cut = nx.minimum_edge_cut(G)
    for e in cut:
        if Pe[e] == 1:
            return "The min cut obtained has a perfectly reliable edge"
    N = int(np.ceil((4*np.log(2/delta)) / (np.prod([1-Pe[tuple(sorted(e))] for e in cut])*(epsilon ** 2))))
    disconnected_count = 0
    for i in range(N):
        if not nx.is_connected(ProbGraph(G,Pe,Pv)):
            disconnected_count += 1
    return 1 - (disconnected_count / N)


def MonteATR(G,p,epsilon,delta):
    """
    Simplified MonteRel to obtain epsilon,delta-estimate of all-terminal reliability of G with edge-operation probabilities p
    """
    c = nx.edge_connectivity(G)
    if c == 0:
        return 0
    N = int(np.ceil((4*np.log(2/delta)) / (((1-p)**c)*(epsilon**2))))
    disconnected_count = 0
    for i in range(N):
        if not nx.is_connected(ProbGraphATR(G,p)):
            disconnected_count += 1
    return 1 - (disconnected_count / N)


def NumberMonteATRTrials(G,p,epsilon,delta):
    """
    In case you'd like to know how many trials MonteATR will run
    """
    c = nx.edge_connectivity(G)
    return int(np.ceil((4*np.log(2/delta)) / (((1-p)**c)*(epsilon**2))))


from pysat.formula import CNF
import pyapproxmc
import subprocess


def nxgraph_to_relnet(G, p):
    """
    Input: networkX graph G and edge-operation probability p
    """
    vxstring = str()
    for v in G.nodes:
        vxstring += " "+str(v)
    edgestring = str()
    for e in G.edges:
        edgestring += "\n"+"e "+str(e[0])+" "+str(e[1])+" "+str(p)
    with open("copy.txt", "w") as file:
        file.write("p g\nT"+vxstring+edgestring)
    subprocess.run('python3 -m relnet copy.txt copy.cnf', shell=True, stdout=subprocess.DEVNULL)


def fast_monteATR(G,p,epsilon=.8,delta=.2):
    nxgraph_to_relnet(G,p)
    cnf = CNF(from_file='copy.cnf')
    c = pyapproxmc.Counter(epsilon = epsilon, delta = delta)
    for i in cnf.clauses:
        c.add_clause(i)
    distring = cnf.to_dimacs()
    distring_s1 = distring.split('p')
    distring_s2 = distring_s1[0].split(' ')
    num_sampling_vars = int(distring_s2[-2])
    count = c.count(projection = range(1,num_sampling_vars+1))
    return 1 - (count[0]*2**count[1]/(2**num_sampling_vars))


def Centroid(S):
    """
    Input: a list of tuples or a numpy array S representing a list of points
    Output: the centroid, or center of mass, of the points in S
    """
    dim = len(S[0])
    L = []
    for j in range(dim):
        L.append(sum([S[i][j] for i in range(len(S))]) / len(S))
    return np.array(L)


def center_of_SEC(R):
    """
    Input: region R from find_regions
    Output: numpy.array point which minimizes the max distance to a point in R.
    """
    vertex_pos = np.array([(pt.x, pt.y) for pt in R])
    best_point = None
    if len(R) > 3:
        min_max_dist = None
        V=Voronoi(vertex_pos,furthest_site=True,qhull_options="QJ")
        # to find the smallest enclosing circle: first, check if it is defined by two points
        # we would like to find the edges of V, check the regions on either side (corresponding to points which might be on the perimeter of SEC). 
        for pt in V.ridge_points:
            q1 = vertex_pos[pt[0]]
            q2 = vertex_pos[pt[1]]
            c = Centroid([q1,q2])
            if max([math.dist(c,vx_pt) for vx_pt in vertex_pos]) == math.dist(c,q1):
                min_max_dist = math.dist(c,q1)
                best_point = c
        # otherwise, the center of the SEC is a vertex of V
        if best_point is None:
            min_max_dist = max([math.dist(V.vertices[0], vx_pt) for vx_pt in vertex_pos])
            for v in V.vertices:
                v_max_dist = max([math.dist(v, vx_pt) for vx_pt in vertex_pos])
                if v_max_dist <= min_max_dist:
                    best_point = v
                    min_max_dist = v_max_dist
    if len(R) == 3:
        min_max_dist = None
        c = Centroid(vertex_pos)
        vertex_pos = np.vstack([vertex_pos, c])
        V=Voronoi(vertex_pos,furthest_site=True,qhull_options="QJ")
        for pts in V.ridge_points:
            q1 = vertex_pos[pts[0]]
            q2 = vertex_pos[pts[1]]
            c = Centroid([q1,q2])
            if max([math.dist(c, vx_pt) for vx_pt in vertex_pos]) == math.dist(c,q1):
                min_max_dist = math.dist(c,q1)
                best_point = c
        if best_point is None:
            min_max_dist = max([math.dist(V.vertices[0], vx_pt) for vx_pt in vertex_pos])
            for v in V.vertices:
                v_max_dist = max([math.dist(v, vx_pt) for vx_pt in vertex_pos])
                if v_max_dist <= min_max_dist:
                    best_point = v
                    min_max_dist = v_max_dist
    if len(R) == 2:
        best_point = Centroid(vertex_pos)
    if len(R) == 1:
        best_point = np.array([R[0].x, R[0].y])
    return best_point


'''
The following function is the basis for the simulations in "Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms" and "Node Placement to Maximize Reliability of a Communication Network with Application to Satellite Swarms."

Note that we obtained estimates of the all-terminal reliability for our application. The function can be modified to use the function MonteRel instead if vertices in the desired application are not perfectly reliable.
'''


def best_new_nbrhd(G, radius, p, epsilon, delta, buffer=None):
    """
    Returns a pair of most reliable neighborhood and the resulting reliability.
    """
    if buffer == None:
      n = G.order
      rels = []
      regs = find_regions(points(G), radius, maximal_only=True)
      for R in regs:
        H = G.copy()
        H.add_node(n)
        for i in [v.label for v in R]:
          H.add_edge(i,n)
        relH = fast_monteATR(H,p,epsilon,delta)
        rels.append(relH)
      best_rel = rels.index(max(rels))
      return (regs[best_rel], max(rels))
    else:
      regs = find_buffer_regions_max(points(G), radius, buffer)
      regs_points = [reg[0][0] for reg in regs]
      n = len(G.nodes)
      rels = []
      for N in regs_points:
        H = G.copy()
        H.add_node(n)
        for i in [v.label for v in N]:
          H.add_edge(i,n)
        # relH = MonteATR(H,p,epsilon,delta)
        relH = fast_monteATR(H,p,epsilon,delta)
        rels.append(relH)
      if len(rels) > 0:
          best = rels.index(max(rels))
          return (regs[best], max(rels))
      else:
          return None


def best_new_location(G,r,p,epsilon,delta):
    """
    Returns best position for a new vertex added to G to maximize all-terminal reliability
    """
    best_nbrhd = best_new_nbrhd(G,radius=r,p=p,epsilon=epsilon,delta=delta)[0]
    return center_of_SEC(best_nbrhd)


def best_new_formation(G,r,p,epsilon,delta):
    '''
    Obtain a new unit disk graph by adding a vertex to G to maximize (all-terminal) reliability.
    '''
    n = len(G.nodes)
    pos = {i : G.nodes[i]['pos'] for i in range(n)}
    pos[n] = best_new_location(G,r,p,epsilon,delta)
    return nx.random_geometric_graph(n+1,r,pos=pos)


def best_move(G,r,p,epsilon,delta):
    '''
    Obtain a new unit disk graph from G by moving a single vertex to a location which maximizes (all-terminal) reliability.
    '''
    n = len(G.nodes)
    L = []
    for i in range(n):
        induced_pos = dict()
        for h in range(i):
            induced_pos[h] = G.nodes[h]['pos']
        for h in range(i+1,n):
            induced_pos[h-1] = G.nodes[h]['pos']
        H = nx.random_geometric_graph(n-1,r,pos=induced_pos)
        L.append(best_new_formation(H,r,p,epsilon,delta))
    relsL = [MonteATR(l,p,epsilon,delta) for l in L]
    for i in range(n):
        if relsL[i] == max(relsL):
            return (L[i], relsL[i])

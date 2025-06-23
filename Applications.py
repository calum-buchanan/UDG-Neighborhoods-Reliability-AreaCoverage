'''
Functions which can be used to recreate the simulations sections of "Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms" and "Node Placement to Maximize Reliability of a Communication Network with Application to Satellite Swarms" by Calum Buchanan, Puck Rombach, James Bagrow, and Hamid R. Ossareh.

Note: opening 'UDG-Neighborhoods-and-Reliability.py' and 'UDG-Layouts.py' imports necessary packages for these functions

Note: our simulations involved all-terminal reliability rather than the more general case with unreliable vertices. Small changes can be made to the functions in UDG-Neighborhoods-and-Reliability.py to account for the general case.
'''


exec(open('UDG-Neighborhoods-and-Reliability.py').read())
exec(open('UDG-Layouts.py').read())
# Also use:
# Smallest enclosing circle - Library (Python)
# Copyright (c) 2020 Project Nayuki
# https://www.nayuki.io/page/smallest-enclosing-circle
exec(open('smallest_enclosing_circle.py').read())


from scipy.spatial import ConvexHull, convex_hull_plot_2d



def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0


"""
Benchmarking for preliminary paper:
CB, James Bagrow, Puck Rombach, Hamid R. Ossareh.
"Node Placement to Maximize Reliability of a Communication Network with Application to Satellite Swarms"
In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
"""

def add_random_vx(G,r):
    n = len(G.nodes)
    A = [G.nodes[i]['pos'] for i in G.nodes]
    minX = min([A[i][0] for i in range(G.order())])
    maxX = max([A[i][0] for i in range(G.order())])
    minY = min([A[i][1] for i in range(G.order())])
    maxY = max([A[i][1] for i in range(G.order())])
    hull = np.array(A)
    while len(A) < n+1:
        p = np.array([(rng.uniform(minX, maxX), rng.uniform(minY, maxY))])
        if in_hull(p,hull):
            A.append([p[0][0],p[0][1]])
    pos = {i : A[i] for i in range(n+1)}
    return nx.random_geometric_graph(n+1,r,pos = pos)


def num_additions_to_best_rel(G,r,p,epsilon,delta):
    n = len(G.nodes)
    best = best_new_nbrhd(G,radius=r,p=p,epsilon=epsilon,delta=delta)
    while MonteATR(G,p,epsilon,delta) <= best[1]:
        G = add_random_vx(G,r)
    return len(G.nodes) - n


def comparison_stats(n,r,p,epsilon,delta,trials):
    L = []
    while len(L)<trials:
        G = nx.random_geometric_graph(n,r)
        if nx.is_connected(G):
            L.append(G)
    rels = []
    for l in L:
        rel_l = MonteATR(l,p,epsilon,delta)
        lminus = l.copy()
        lminus.remove_node(n-1)
        newp = find_regions(points(lminus),r,maximal_only=True)
        bestpos = best_new_nbrhd(lminus,radius=r,p=p,epsilon=epsilon,delta=delta)
        rels.append((l, bestpos[0], (rel_l, bestpos[1])))
    return rels


def benchmark(n,r,p,epsilon,delta,trials):
    L = []
    while len(L) < trials:
        G = nx.random_geometric_graph(n,r)
        if nx.is_connected(G):
            L.append(G)
    numbers = []
    for l in L:
        num = num_additions_to_best_rel(l,r,p,epsilon,delta)
        numbers.append(num)
        print(num)
    return numbers




'''
Applications regarding area coverage and reliability --
Benchmarking for manuscript "Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms", CB, Puck Rombach, James Bagrow, Hamid R. Ossareh.
'''

'''
LARGEST EMPTY CIRCLE ALGORITHM

Input: A dictionary or list of points (like the data class)
Output: A pair whose first coord is the center of the largest 
empty circle whose center is contained in the convex hull of 
'points' and whose second coord is the radius of this circle.
'''
def LEC(points):
    vertex_pos = np.array([(points[i].x, points[i].y) for i in range(len(points))])
    hull = ConvexHull(vertex_pos)
    edges = [sorted(hull.simplices[i]) for i in range(len(hull.simplices))]
    best_point = None
    max_dist = 0
    V = Voronoi(vertex_pos)
    for v in V.vertices:
        if in_hull(v, vertex_pos):
            d = min([math.dist(v, w) for w in vertex_pos])
            if d > max_dist:
                max_dist = d
                best_point = v
    # Check intersection of ridges with edges of convex hull
    for a in V.ridge_points:
        for b in hull.simplices:
            if sorted(a) != sorted(b):
                # find point of intersection between the line between a[0] and a[1] and convex hull
                x1a = vertex_pos[a[0]][0]
                y1a = vertex_pos[a[0]][1]
                x2a = vertex_pos[a[1]][0]
                y2a = vertex_pos[a[1]][1]
                m_a = (x2a - x1a) / (y2a - y1a)
                c_a = Centroid([(x1a, y1a), (x2a, y2a)])
                # line_a = m_a (x - c_a[0]) - y - c_a[1]
                x1b = vertex_pos[b[0]][0]
                y1b = vertex_pos[b[0]][1]
                x2b = vertex_pos[b[1]][0]
                y2b = vertex_pos[b[1]][1]
                m_b = (y2b - y1b) / (x2b - x1b)
                # line_b = m_b (x - x1b) - y + y1b
                x = (-m_b*x1b + y1b + m_a*c_a[0] - c_a[1]) / (m_a - m_b)
                y = m_b * x - m_b * x1b + y1b
                if x1b <= x <= x2b or x2b <= x <= x1b:
                    d = min([math.dist((x,y), w) for w in vertex_pos])
                    if d > max_dist:
                        max_dist = d
                        best_point = v
    return (best_point, max_dist)


def UDG_ngon_pos(n0: float):
    '''
    Dictionary of vertex positions for regular n0-gon with side-lengths 0.9
    '''
    n = n0
    d = .9 / (2 * float("%.16f" % math.sin(math.pi / n)))
    return {k : (d * float("%.16f" % math.cos((k+1) * 2 * math.pi / n)), d * float("%.16f" % math.sin((k+1) * 2 * math.pi / n))) for k in range(n)}


def fill_ngon(n0: float, iterations: float, p: float, epsilon: float, delta: float):
    """
    Fill in region surrounded by a regular n-gon UDG with edge-lengths 0.9
    Return: list of graphs obtained by adding a vx to maximize reliability at each iteration, plus the last graph with spring layout at the end.
    """
    ngon_pts = UDG_ngon_pos(n0)
    r = 1
    n = n0
    G = nx.random_geometric_graph(n, radius=r, pos=ngon_pts)
    node_pos = ngon_pts
    graphs_list = [ G ]
    rels_list = [ (n * (p ** (n-1)) * (1-p)) + (p ** n) ]
    for it in range(iterations):
        #Determine most reliable addition
        best_reg = best_new_nbrhd(G,radius=r,p=p,epsilon=epsilon,delta=delta)
        R = [(r.x, r.y) for r in best_reg[0]]
        #pt = center_of_SEC(best_reg[0])
        circle = make_circle(R)
        pt = np.array([circle[0], circle[1]])
        #Add vertex to G
        node_pos[n] = pt
        Gx = nx.random_geometric_graph(n+1,radius=r,pos=node_pos)
        graphs_list.append(Gx)
        rels_list.append(best_reg[1])
        n += 1
        G = Gx
        # save_geo(G, 'ngon_springatend_iter'+str(it))
    new_node_pos = spring_layout_tempfix(G, k=None, pos=node_pos, fixed=[i for i in range(n0)], iterations=50, threshold=1e-4, weight="weight", scale=1, dim=2)
    G_spring = nx.random_geometric_graph(n,radius=r,pos=new_node_pos)
    graphs_list.append(G_spring)
    return graphs_list, rels_list


def fill_ngon_springiters(n0: float, iterations: float, p: float, epsilon: float, delta: float):
    """
    Fill in region surrounded by a regular n-gon UDG with edge lengths 0.9.
    Return two lists of graphs, each starting with a cycle of length n0, formed as follows.
    Add a vertex to the most recently obtained graph to maximize reliability. Add this graph to the former list.
    Perform the spring layout algorithm to the current graph, and add this to the latter list.
    """
    ngon_pts = UDG_ngon_pos(n0)
    r = 1
    n = n0
    G = nx.random_geometric_graph(n, radius=r, pos=ngon_pts)
    """
    Maximize reliability, then spring layout at each step
    """
    node_pos = ngon_pts
    graphs_list = [ G ]
    rels_list = [ (n * (p ** (n-1)) * (1-p)) + (p ** n) ]
    for it in range(iterations):
        #Determine most reliable addition
        best_reg = best_new_nbrhd(G,radius=r,p=p,epsilon=epsilon,delta=delta)
        pt = center_of_SEC(best_reg[0])
        #Add vertex to G
        node_pos[n] = pt
        Gx = nx.random_geometric_graph(n+1,radius=r,pos=node_pos)
        new_node_pos = spring_layout_tempfix(Gx, k=None, pos=node_pos, fixed=[i for i in range(n0)], iterations=50, threshold=1e-4, weight="weight", scale=1, dim=2)
        Gx_spring = nx.random_geometric_graph(n+1,radius=r,pos=new_node_pos)
        graphs_list.append(Gx_spring)
        rels_list.append(best_reg[1])
        node_pos = new_node_pos
        n += 1
        G = Gx_spring
        # save_geo(Gx_spring, str(it))
    return graphs_list, rels_list


def fill_ngon_buffer(n0: float, buffer: float, iterations: float, p: float, epsilon: float, delta: float):
    '''
    Fill in region surrounded by a regular n-gon UDG with edge lengths 0.9 by adding vertices one at a time to maximize reliability under the constraint that no other vertices be within buffer distance.
    '''
    node_pos = UDG_ngon_pos(n0)
    r = 1
    G = nx.random_geometric_graph(n0,radius=r,pos=node_pos)
    n = n0
    rminus = r - .01
    graphs_list = [ G ]
    rels_list = [ (n * (p ** (n-1)) * (1-p)) + (p ** n) ]
    for it in range(iterations):
        best = best_new_nbrhd(G, rminus, p, epsilon, delta, buffer=buffer)
        if best is not None:
            rels_list.append(best[1])
            node_pos[n] = best[0][1]
            Gx = nx.random_geometric_graph(n+1,radius=r,pos=node_pos)
            graphs_list.append(Gx)
            G = Gx
            n += 1
        else:
            break
    d = LEC(perturb_points(points(Gx), .001))
    return graphs_list, rels_list, d


def fill_ngon_buffer_springiters(n0: float, buffer: float, iterations: float, p: float, epsilon: float, delta: float, k=None):
    '''
    A combination of fill_ngon_springiters and fill_ngon_buffer. First, add vertices one at a time to maximize reliability under buffer constraint, and apply UDG spring layout algorithm each time a vertex is added.
    '''
    ngon_pts = UDG_ngon_pos(n0)
    node_pos = ngon_pts
    r = 1
    G = nx.random_geometric_graph(n0,radius=r,pos=node_pos)
    n = n0
    rminus = r - .01
    graphs_before_spring = [ G ]
    graphs_after_spring = [ G ]
    rels_list = [ (n * (p ** (n-1)) * (1-p)) + (p ** n) ]
    for it in range(iterations):
        best = best_new_nbrhd(G, rminus, p, epsilon, delta, buffer=buffer)
        if best is not None:
            rels_list.append(best[1])
            node_pos[n] = best[0][1]
            Gx = nx.random_geometric_graph(n+1,radius=r,pos=node_pos)
            graphs_before_spring.append(Gx)
            new_node_pos = spring_layout_tempfix(Gx, k=k, pos=node_pos, fixed=[i for i in range(n0)], iterations=50, threshold=1e-4, weight="weight", scale=1, dim=2)
            Gx_spring = nx.random_geometric_graph(n+1, radius = r, pos = new_node_pos)
            graphs_after_spring.append(Gx_spring)
            node_pos = new_node_pos
            G = Gx_spring
            n += 1
        else:
            break
    d = LEC(perturb_points(points(G), .001))
    return graphs_before_spring, graphs_after_spring, rels_list, d


def fill_ngon_randomly_spring(n0: float, iterations: float, p: float, epsilon: float, delta: float, k=None):
    '''
    Add vertices randomly one at a time to the interior of an n-gon, conditioned on the resulting graph being connected. Perform the UDG spring layout algorithm at each iteration. Also, record reliabilities.
    '''
    ngon_pts = UDG_ngon_pos(n0)
    A = [ngon_pts[i] for i in range(n0)]
    minX = min([A[i][0] for i in range(n0)])
    maxX = max([A[i][0] for i in range(n0)])
    minY = min([A[i][1] for i in range(n0)])
    maxY = max([A[i][1] for i in range(n0)])
    hull = np.array(A)
    r = 1
    n = n0
    G = nx.random_geometric_graph(n, radius=r, pos=ngon_pts)
    graphs_list = [ G ]
    graphs_list_nospring = [ G ]
    rels_list = [ (n * (p ** (n-1)) * (1-p)) + (p ** n) ]
    while n < n0 + iterations:
        pt = np.array([(rng.uniform(minX, maxX), rng.uniform(minY, maxY))])
        if in_hull(pt,hull) and min([math.dist(pt[0],v) for v in A]) < 1:
            A.append((pt[0][0],pt[0][1]))
            n += 1
            next_pos = {i : A[i] for i in range(n)}
            Gx = nx.random_geometric_graph(n, r, pos=next_pos)
            graphs_list_nospring.append(Gx)
            rels_list.append(MonteATR(Gx, p, epsilon, delta))
            spring_pos = spring_layout_tempfix(Gx, k, pos=next_pos, fixed=[i for i in range(n0)], iterations=50, threshold=1e-4, weight="weight", scale=1, dim=2)
            G = nx.random_geometric_graph(n, r, pos=spring_pos)
            graphs_list.append(G)
    d = LEC(perturb_points(points(Gx), .001))
    dspring = LEC(perturb_points(points(G), .001))
    return graphs_list_nospring, graphs_list, rels_list, d, dspring

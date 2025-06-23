'''
Adapted from networkx.drawing.layout (https://networkx.org/documentation/stable/_modules/networkx/drawing/layout.html)
Adapted functions: spring_layout and _fruchterman_reingold
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
from networkx.utils import np_random_state

plt.rcParams["figure.figsize"] = [4,4]



"""
First, a few helpful functions for plotting or saving a unit
disk graph and for depicting a vertex for every possible 
neighborhood (see Reliability.py)
"""

def plot_geo(G):
    node_pos={i : G.nodes[i]['pos'] for i in range(G.order())}
    nx.draw(G, pos=node_pos, node_size=75)
    plt.show()


def save_geo(G, name: str):
    node_pos={i : G.nodes[i]['pos'] for i in range(G.order())}
    nx.draw(G, pos=node_pos, node_size=75)
    plt.savefig(name+'.pdf')
    plt.close()


def plot_new_positions(G,r,NewP):
    n=len(G.nodes)
    k=len(NewP)
    node_pos={}
    for i in range(n):
        node_pos[i]=G.nodes[i]["pos"]
    for i in range(k):
        node_pos[n+i]=NewP[i][0]
    Gx=nx.random_geometric_graph(n+k,radius=r,pos=node_pos)
    colors=['blue' if i<n else 'red' for i in range(n+k)]
    nx.draw(Gx,node_color=colors,pos=node_pos)
    plt.show()

'''
Now, the modified Fruchterman-Reingold algorithm
'''

def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph
    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


@np_random_state(10)
def spring_layout_tempfix(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    radius=1,
):
    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}
    A = nx.to_numpy_array(G, weight=weight)
    if k is None and fixed is not None:
        # We must adjust k by domain size for layouts not near 1x1
        nnodes, _ = A.shape
        k = dom_size / np.sqrt(nnodes)
    pos = _fruchterman_reingold_tempfix(
        A, k, pos_arr, fixed, iterations, threshold, dim, seed
    )
    pos = dict(zip(G, pos))
    return pos


@np_random_state(7)
def _fruchterman_reingold_tempfix(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None, radius=1
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # matrix of difference between points
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # distance between points
    distance = np.linalg.norm(delta, axis=-1)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum("ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        new_pos = pos + delta_pos
        new_delta = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        new_distance = np.linalg.norm(new_delta, axis=-1)
        if not any([A[i][j]!=0 and new_distance[i][j]>=1 for i in range(nnodes) for j in range(i)]):
            pos = new_pos
            delta = new_delta
            distance = new_distance
        t -= dt
        # if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            # break
    return pos


@np_random_state(10)
def spring_layout_onebyone_tempfix(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    radius=1,
    perm=False
):
    import numpy as np

    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}
    A = nx.to_numpy_array(G, weight=weight)
    if k is None and fixed is not None:
        # We must adjust k by domain size for layouts not near 1x1
        nnodes, _ = A.shape
        k = dom_size / np.sqrt(nnodes)
    if perm is True:
        pos = _fruchterman_reingold_onebyone_tempfix_perm(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed, radius
        )
    pos = _fruchterman_reingold_onebyone_tempfix(
        A, k, pos_arr, fixed, iterations, threshold, dim, seed, radius
    )
    pos = dict(zip(G, pos))
    return pos


@np_random_state(7)
def _fruchterman_reingold_onebyone_tempfix_perm(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None, radius=1
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # matrix of difference between points
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # distance between points
    distance = np.linalg.norm(delta, axis=-1)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        perm = seed.permutation(len(A[0]))
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum("ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        # move vertices one at a time
        new_pos = pos.copy()
        for i in perm:
            new_pos[i] = pos[i] + delta_pos[i]
            new_delta = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
            new_distance = np.linalg.norm(new_delta, axis=-1)
            if any([A[i][j]!=0 and new_distance[i][j]>=radius for j in range(nnodes)]):
                new_pos[i] = pos[i]
        pos = new_pos
        delta = new_delta
        distance = new_distance
        t -= dt
        # if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            # break
    return pos

@np_random_state(7)
def _fruchterman_reingold_onebyone_tempfix(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None, radius=1
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # matrix of difference between points
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # distance between points
    distance = np.linalg.norm(delta, axis=-1)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum("ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        # move vertices one at a time
        not_fixed = [v for v in range(len(A[0])) if v not in fixed]
        s = seed.choice(len(not_fixed))
        u = not_fixed[s]
        new_pos = pos.copy()
        new_pos[u] = pos[u] + delta_pos[u]
        new_delta = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        new_distance = np.linalg.norm(new_delta, axis=-1)
        if any([A[u][j]!=0 and new_distance[u][j]>=radius for j in range(nnodes)]):
            new_pos[u] = pos[u]
        pos = new_pos
        delta = new_delta
        distance = new_distance
        t -= dt
        # if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            # break
    return pos

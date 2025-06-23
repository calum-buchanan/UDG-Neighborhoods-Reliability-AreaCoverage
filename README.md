
# Python code to enumerate regions of intersection formed by a set of circles or tori of equal radii, with application to reliability and area coverage in unit disk graphs

This code was used to produce the simulations for the paper *Vertex addition to a ball graph with application to reliability and area coverage in autonomous swarms* by Calum Buchanan, Puck Rombach, James Bagrow, and Hamid R. Ossareh (2025).
It includes the functions used for our preliminary conference paper *Node Placement to Maximize Reliability of a Communication Network with Application to Satellite Swarms* in the proceedings of the 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Honolulu, Oahu, HI, USA.
It also includes a modified Fruchterman-Reingold algorithm which avoids destroying edges in a unit disk graph.

## Python 3 Requirements:

```python
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
from scipy.spatial import ConvexHull, convex_hull_plot_2d
```

The file `UDG-Neighborhoods-and-Reliability.py` contains a function to find the center of the smallest enclosing circle for a finite set of points. This can also be done using the function `make_circle` in `smallest_empty_circle.py` from https://www.nayuki.io/page/smallest-enclosing-circle. We use this function in one instance in `Applications.py`.

To obtain relative estimates of reliability, one can use the function MonteRel provided in `UDG-Neighborhoods-and-Reliability.py`. To find the all-terminal reliability of a highly reliable graph, we instead use RelNet from https://github.com/meelgroup/RelNet. We then have additional requirements:

```python
from pysat.formula import CNF
import pyapproxmc
import subprocess
```

## Example use:

```python
>>> exec(open('Applications.py').read())
>>> n0 = 15
>>> it = 15
>>> p = .9
>>> buffer = .65
>>> epsilon = .8
>>> delta = .2
>>> fill_ngon_buffer(n0, buffer, it, p, epsilon, delta)
```

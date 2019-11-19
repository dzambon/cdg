# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements the generation of Delaunay-triangulation graphs.
# --------------------------------------------------------------------------------
from collections import OrderedDict

import numpy as np
from .graph import Graph

def generate_delaunay_adjacency(points):
    from scipy.spatial import Delaunay as scipy_del
    
    list_of_points = points if points.ndim == 3 else [points]
    no_vertices = list_of_points[0].shape[0]

    list_of_adjmat = []
    for p in list_of_points:
        # Delaunay triangulation
        tri = scipy_del(p)
        # Adjacency matrix from triangulation
        adj_matrix = np.zeros((no_vertices, no_vertices))
        for t in tri.simplices:
            for i in range(0, 3):
                j = np.mod(i + 1, 3)
                adj_matrix[t[i], t[j]] = 1
                adj_matrix[t[j], t[i]] = 1
        list_of_adjmat.append(adj_matrix)
    return list_of_adjmat


class DelaunayGraphs(object):
    
    def get(self, seed_points=10, classes=20, no_graphs=10, sigma=.3, include_seed_graph=True):
        """
        Generate a data set of Delaunay's triangulation graphs.
        :param seed_points: If `np.array` (memory_order, no_points, 2) seed points for the graph generating mechanism.
            If `int` then no_points=seed_points points are created (def = 10)
        :param classes: If `list` then it is a list of class identifiers. If `int` then all classes from 0
            to `classes` are created. Class identifiers are nonnegative integers: `id = 0, 1, 2, ...` are
            all admissible classes. Class 0 is usually intended as reference class. As `id` increases,
            class `id` get 'closer' to class 0. (def = 20)
        :param no_graphs: number of graphs to be generated. If `int` every class will have the same
            number of graphs, otherwise it can be a dictionary {classID: no graphs} (def = 10)
        """
        
        # parse classes
        if isinstance(classes, list):
            self.classes = classes.copy()
        elif isinstance(classes, int):
            self.classes = [i for i in range(classes + 1)]
        
        # parse no_graphs
        if include_seed_graph:
            no_graphs -= 1
        if isinstance(no_graphs, int):
            no_graphs_dict = {c: no_graphs for c in self.classes}
        else:
            no_graphs_dict = no_graphs

        # parse seed_points
        scale = 10.
        if type(seed_points) is int:
            self.seed_points = np.random.rand(seed_points, 2) * scale
        else:
            self.seed_points = seed_points.copy()
        assert self.seed_points.shape[1] == 2, "The point dimension must be 2."
        
        no_points = self.seed_points.shape[0]
        graphs_list = OrderedDict()
        for c in self.classes:

            # update radius
            radius = scale * (2./3.)**(c-1) if c>0 else 0
            
            # update support points
            support_points = self.seed_points.copy()
            phase = np.random.rand(no_points) * 2 * np.pi
            support_points[:, 0] += radius * np.sin(phase)
            support_points[:, 1] += radius * np.cos(phase)

            # create graphs from support points
            new_points = support_points[None, ...] +  np.random.randn(no_graphs_dict[c], no_points, 2) * sigma
            new_adjmat = generate_delaunay_adjacency(new_points)
            graphs_list[c] = []
            if include_seed_graph:
                graphs_list[c] += [Graph(generate_delaunay_adjacency(support_points)[0], support_points, None)]
            graphs_list[c] += [Graph(new_adjmat[i], new_points[i], None) for i in range(no_graphs_dict[c])]

        return graphs_list

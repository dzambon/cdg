# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements (wrappers of) graph distances.
# --------------------------------------------------------------------------------
from cdg.graph import *

class FrobeniusGraphDistance(Distance):
    '''
    Distance for graphs with identified nodes:
    d(g1, g2) = | g1.adj - g2.adj|_F + | g1.nf - g2.nf|_F + | g1.ef - g2.ef|_F.
    '''
    name = 'frobenius norm'
    
    def _distance(self, source, target, symmetric, paired, verbose):
        assert isinstance(source[0], cdg.graph.Graph), 'graph format is not cdg.graph.Graph'
        return self._handle_single_measure(source=source, target=target,
                                           single_measure_fun=self._single_distance,
                                           symmetric=symmetric, paired=paired, verbose=verbose)
    
    def _single_distance(self, g1, g2, verbose=False):
        return np.linalg.norm(np.logical_xor(g1.adj, g2.adj)) \
               + np.linalg.norm(g1.nf - g2.nf) \
               + np.linalg.norm(g1.ef - g2.ef)


class GraphEditDistanceNX(Distance):
    '''
    Graph edit distance, as provided by `networkx`.
    '''
    name = 'nxGED'
    
    node_cost = None
    edge_cost = None
    
    def __init__(self, node_cost=None, edge_cost=None, n_jobs=1):
        self.node_cost = self.get_cost(node_cost)
        self.edge_cost = self.get_cost(edge_cost)
        self.n_jobs = n_jobs
    
    def get_cost(self, cost=None):
        if cost == 'euclidean':
            return self.euclidean_cost
        else:
            return cost
    
    def _distance(self, source, target, symmetric, paired, verbose):
        from networkx import Graph as nx_Graph
        assert isinstance(source[0], nx_Graph), 'graph not in networkx format'
        return self._handle_single_measure(source=source, target=target,
                                           single_measure_fun=self._single_distance,
                                           symmetric=symmetric, paired=paired, verbose=verbose)
    
    def _single_distance(self, g1, g2, verbose=False):
        from networkx import graph_edit_distance
        return graph_edit_distance(g1, g2,
                                   node_subst_cost=self.node_cost,
                                   edge_subst_cost=self.edge_cost)
    
    @staticmethod
    def euclidean_cost(o1, o2):
        from scipy.spatial.distance import euclidean
        return euclidean(o1['vec'], o2['vec'])

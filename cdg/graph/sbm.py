# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Implements models derived by the degree-corrected stochastic block model.
# --------------------------------------------------------------------------------
import numpy as np
from .graph import Graph
from functools import reduce

class DegreeCorrectedStochasticBlockModel(object):
    """
    Wilson, James D., Nathaniel T. Stevens, and William H. Woodall. ``Modeling and estimating change in temporal
    networks via a dynamic degree corrected stochastic block model.'' arXiv preprint arXiv:1605.04049 (2016).
    """
    
    def __init__(self, communities, prob_matrix, theta=None, delta=None):
        """
        :param communities: (list of lists) partition of set {0, ..., n-1}
        :param prob_matrix: (np.ndarray(no_communities, no_communities)) inter- and intra-community link propensity
        :param theta: (np.ndarray(no_nodes,)) degree correction
        :param delta: (np.ndarray(no_communities,)) parameters to generate random theta
        """
        
        # Nodes
        node_set = sorted(reduce((lambda x, y: x + y), communities))
        assert (np.arange(len(node_set)) == np.array(
            node_set)).all(), "communities is not a partition of {0, 1, ..., n-1}"
        self.no_vertices = len(node_set)
        # Communities
        self.communities = [np.array(c) for c in communities]
        self.no_communities = len(self.communities)
        membership_onehot = np.zeros((self.no_vertices, self.no_communities), dtype=int)
        for ci, c in enumerate(self.communities):
            membership_onehot[c, ci] = 1
        # Community link probabilities
        self.probabilities = prob_matrix if isinstance(prob_matrix, np.ndarray) else np.array(prob_matrix)
        assert self.probabilities.ndim == 2
        assert self.probabilities.shape[0] == self.probabilities.shape[1]
        assert self.probabilities.shape[0] == self.no_communities
        assert (self.probabilities == self.probabilities.T).all()
        # Degree corrections
        self.theta = theta if theta is not None else self._generate_theta(delta)
        Theta_mat = np.dot(self.theta.reshape(-1, 1), self.theta.reshape(1, -1))
        # Expected adjacency matrix
        self.expected_adj = membership_onehot.dot(self.probabilities).dot(membership_onehot.T)
        self.expected_adj *= Theta_mat

    def _generate_theta(self, delta):
        """ Generates theta from delta. """
        theta = np.ones(self.no_vertices)
        if delta is None:
            pass
        else:
            delta_ar = delta if isinstance(delta, np.ndarray) else np.array(delta)
            if delta_ar.ndim == 1:
                delta_ar = delta_ar.reshape(-1, 1)
                delta_ar = np.hstack([delta_ar] * 2)
            elif delta_ar.shape[1] < 2:
                delta_ar = np.vstack([delta_ar] * 2)
            
            assert delta_ar.ndim ==2 and delta_ar.shape[1] == 2
            assert np.all(delta_ar >= 0.)

            for r, cr in enumerate(self.communities):
                theta[cr] += np.random.uniform(low=-delta_ar[r, 0], high=delta_ar[r, 1], size=len(cr))
                theta[cr] *= len(cr) / np.sum(theta[cr])
        return theta
    
    def get(self, no_graphs=1, distrib="poisson", format="cdg"):
        """
        Generates a set of graphs from the model.
        :param no_graphs: (int)
        :param distrib: (str in {"poisson", "uniform"})
        :param format: (str, def="cdg")
        :return: a list of `no_graphs` instances of cdg.graph.Graph
        """
        if distrib == "poisson":
            rand_vals = np.random.poisson(lam=self.expected_adj,
                                          size=(no_graphs, self.no_vertices, self.no_vertices))
        else:
            rand_vals = np.random.rand(no_graphs, self.no_vertices, self.no_vertices)

        for n in range(no_graphs):
            rand_vals[n] = np.tril(rand_vals[n], -1) + np.tril(rand_vals[n], -1).T + np.eye(self.no_vertices)

        if distrib == "poisson":
            adjmat = rand_vals > 0
            ef = rand_vals[..., None]
        else:
            adjmat = rand_vals <= self.expected_adj[None, ...]
            ef = [None] * no_graphs

        adjmat = adjmat.astype(int)
        nf = [None] * no_graphs
        
        if format == 'npy':
            return adjmat, nf, ef
        else:
            return [Graph(adjmat[i], nf[i], ef[i]) for i in range(no_graphs)]
        

class StochasticBlockModel(DegreeCorrectedStochasticBlockModel):
    """
    P. W. Holland, K. B. Laskey, and S. Leinhardt, ``Stochastic blockmodels: First steps'', Social Networks, vol. 5,
    no. 2, pp. 109–137, 19
    """
    
    def __init__(self, communities, prob_matrix):
        super().__init__(communities=communities,
                         prob_matrix=prob_matrix,
                         delta=None)

    def get(self, no_graphs=1, distrib="uniform", format="cdg"):
        return super().get(no_graphs=no_graphs, distrib=distrib, format=format)


class ErdosRenyiModel(StochasticBlockModel):
    
    def __init__(self, no_vertices, prob_edge):
        super().__init__(communities=[list(range(no_vertices))], prob_matrix=[[prob_edge]])


class DynamicsGenerator(object):
    
    def __init__(self, alpha, getter):
        assert alpha <= 1. and alpha > 0
        self.alpha = alpha    # Continuity parameter
        self.getter = getter

    def get(self, graph_seed=None, no_graphs=1):
        """
        Generates a set of graphs from the model.
        :param no_graphs: (int)
        :param graph_seed: (cdg.Graph, def=False)
        :return: a list of `no_graphs` instances of cdg.graph.Graph
        """
        graph_0 = self.getter() if graph_seed is None else graph_seed
        G = [graph_0]

        for _ in range(no_graphs - 1):
            rand_mat = np.random.rand(*(graph_0.adj.shape)) / 2.
            rand_mat += rand_mat.T
            mask_e = np.where(rand_mat > self.alpha)
            mask_n = np.where(np.diag(rand_mat) > self.alpha)
    
            graph_1 = self.getter()
            graph_1.adj[mask_e] = graph_0.adj[mask_e]
            graph_1.ef[mask_e] = graph_0.ef[mask_e]
            graph_1.nf[mask_n] = graph_0.nf[mask_n]
            
            G.append(graph_1)
            graph_0 = graph_1
        
        return G

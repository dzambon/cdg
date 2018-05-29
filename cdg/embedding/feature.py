# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Implements a single feature extraction in the framework of the other embedding
# techniques.
#
#
# References:
# ---------
# [ijcnn18]
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   Anomaly and Change Detection in Graph Streams through Constant-Curvature
#   Manifold Embeddings.
#   IEEE International Joint Conference on Neural Networks (2018).
#
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 11/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from tqdm import tqdm

import cdg.util.logger
import cdg.util.errors
import cdg.util.prototype
import cdg.embedding.embedding


class GraphFeature(cdg.embedding.embedding.Embedding):
    '''
    Mostly conceptual class. Represents a feature extracted directly from each graph.
    '''
    _name = 'GenericFeature'

    def set_parameters(self, **kwargs):
        super().set_parameters(ed=1, **kwargs)

    def predict(self, graph_list, dataset):
        ct = 0
        x = []
        for graph in tqdm(graph_list, desc='Computing graph features'):
            el = dataset.get_name_and_class(graph)
            if el is None:
                raise ValueError("something went wrong in retrieving of the graph.")
            graphName = el[1]
            x.append(self._predict_single(graphName, dataset.path))
        return np.array([x]).transpose()


class Density(GraphFeature):
    """
    Density of the graph.
    """
    _name = 'Density'

    def _predict_single(self, graph, path):
        return cdg.graph.graph.Graph(graph, path).get_density()


class SpectralGap(GraphFeature):
    """
    Spectral gap of the graph laplacian.
    """
    _name = 'SpectralGap'

    def _predict_single(self, graph, path):
        return cdg.graph.graph.Graph(graph, path).get_laplacian_spec_gap()


class DistanceGraphMean(GraphFeature, cdg.embedding.embedding.DissimilarityBased):
    _name = 'GraphMean'

    def fit(self, dissimilarity_matrix):
        """
        :param dissimilarity_matrix:
        :return: index of the graph to the closest Frechet mean
        """
        mean_id, val = cdg.util.prototype.mean(dissimilarity_matrix, power=2)
        self.mean_id = mean_id
        return self.mean_id, val

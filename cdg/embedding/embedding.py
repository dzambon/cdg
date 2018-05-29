# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Defines the structure of the embedding classes, in particular the ones based on prototypes and
# dissimilarities.
#
#
# References:
# ---------
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
# [1] 
#   PW, Duin Robert, and Pekalska Elzbieta.
#   Dissimilarity Representation For Pattern Recognition, The: Foundations And Applications.
#   Vol. 64. World scientific, 2005.
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
import cdg.util.errors
import cdg.util.serialise
import cdg.util.prototype


class Embedding(cdg.util.logger.Loggable, cdg.util.serialise.Pickable):
    '''
    Very generic embedding.
    '''

    _name = 'GenericEmbedding'
    # dimension of the vector representation
    embedding_dimension = None

    def __init__(self):
        self.log.debug(str(self) + " created")

    def __str__(self, extra=''):
        return self._name + "(ed" + str(self.embedding_dimension) + extra + ")"

    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - ed : dimension of the embedding representation (required).
        """
        self.embedding_dimension = kwargs.pop('ed')

    def reset(self):
        """ Clean the embedding instance from dependency"""
        pass

    def fit(self):
        raise cdg.util.errors.CDGAbstractMethod()

    def predict(self):
        raise cdg.util.errors.CDGAbstractMethod()


class PrototypeBased(Embedding):
    '''
    Defines the structure of prototype-based embeddings.
    So far it assumes that the prototypes are part of the training set.
    '''

    _name = 'GenericPrototypeBased'
    # number of prototypes
    no_prototypes = None
    # indices of the selected prototypes in the training data
    prototype_indices = None

    # matrix X \in \R^{noPrototypes \times embeddingDimension} = each row a prototype vector; 
    # hence, Xt = each column a prototype vector
    _Xt_prototypes = None
    _XtX_prototypes = None

    @property
    def X_prototypes(self):
        return self._Xt_prototypes.transpose()

    def __init__(self):
        super().__init__()
        self.skip_from_serialising(['X_training'])

    def __str__(self, extra=''):
        return super().__str__(extra=";p" + str(self.no_prototypes))

    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - M : number of prototypes (required).
            - ... and all related to the superclass
        """
        self.no_prototypes = kwargs.pop('M')
        super().set_parameters(**kwargs)

    def reset(self):
        super().reset()
        self._Xt_prototypes = None
        self._XtX_prototypes = None
        self.prototype_indices = None

    def predict(self, y):
        """
        Embeds new data points basing on the dissimilarities with respect to the selected prototypes
        :param y: (no_test_points, no_prototypes) distances d(g_i, r_j) between each graph and the
            prototypes
        :return X: (no_test_points, embedding_dimension) embedded data
        """
        N = y.shape[0]
        if y.shape[1] != self.no_prototypes:
            raise ValueError("dimension mismatch")
        X = np.zeros((N, self.embedding_dimension))
        for n in tqdm(range(N), desc='embedding'):
            X[n, :] = self._embed_single_datum(y[n, :])
        return X

    def _embed_single_datum(self, dissimilarity_representation):
        raise cdg.util.errors.CDGAbstractMethod()

    def _prototype_selection(self):
        raise cdg.util.errors.CDGAbstractMethod()


class DissimilarityBased(Embedding):
    '''
    Defines the structure of embeddings based on a dissimilarity matrix.
    '''

    _name = 'GenericDissimilarityBased'
    # dissimilarity matrix adopted for the training
    _dissimilarity_training = None

    @property
    def _max_training_distance(self):
        """ Just a feature. Find the max distance in the dissimilarities for training."""
        return np.max(self._dissimilarity_training)

    def __init__(self):
        super().__init__()
        self.skip_from_serialising(['_dissimilarity_training', '_X_training'])

    def _set_dissimilarity_matrix(self, dissimilarity_matrix):
        """ Stores the dissimilarity matrix. """
        self._dissimilarity_training = dissimilarity_matrix

    def reset(self):
        super().reset()
        self._dissimilarity_training = None


class DissimilarityRepresentation(PrototypeBased, DissimilarityBased):
    """ Dissimilarity Representation class. See [tnnls17, 1]. """
    _name = 'DR'

    def set_parameters(self, **kwargs):
        """
        :param kwargs:
            - M : number of prototypes (required).
            - ed : dimension of the embedding
        """
        ed = kwargs.pop('ed', kwargs.get('M'))
        # if ed != kwargs.get('M'):
        #     raise ValueError("emb. dimension `ed` has to be the same of the num. of prototypes `M`")
        super().set_parameters(ed=ed, **kwargs)
        self.log.debug("ed{} - M{}".format(self.embedding_dimension, self.no_prototypes))

    def fit(self, dissimilarity_matrix, no_annealing=1):
        self._set_dissimilarity_matrix(dissimilarity_matrix=dissimilarity_matrix)
        _, err = self._prototype_selection(dissimilarity_matrix=dissimilarity_matrix,
                                           no_annealing=no_annealing)
        return err

    def _prototype_selection(self, dissimilarity_matrix, no_annealing=1):
        """

        :param dissimilarity_matrix: (no_training_data, no_training_data) np.array of
            dissimilarities d(g_i,g_j) with g_i,g_j training graphs.
        :param no_annealing: number of repeated experiment with different initial state in order to
            avoid local minima
        :return:
            - indices of the prototype indices,
            - value of the objective function reached
        """
        val = np.inf
        for an in range(0, no_annealing):
            prot_id_tmp, val_tmp = cdg.util.prototype.k_centers(dissimilarity_matrix,
                                                                n_prototypes=self.no_prototypes)
            if val_tmp < val:
                prot_id = prot_id_tmp.copy()
                val = val_tmp
        self.prototype_indices = prot_id.copy()
        return self.prototype_indices, val

    @classmethod
    def sample_mean(cls, X, **kwargs):
        """
        :param X: np.array (no_datapoints, embedding_dimension) of which the mean is computed
        :param kwargs: not used
        :return: sample mean
        """
        return np.mean(X, axis=0)

    def _embed_single_datum(self, x):
        """ The point is simply returned as it is. """
        return x

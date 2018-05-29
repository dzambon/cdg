# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Tests in [ijcnn18]. This file has to be placed in `cdg/simulation/` folder.
#
# References:
# ---------
# [ijcnn18]
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   Anomaly and Change Detection in Graph Streams through Constant-Curvature
#   Manifold Embeddings.
#   IEEE International Joint Conference on Neural Networks (2018).
#
#
# -----------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# ------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 26/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cdg.simulation.simulations


class SimulationGraphSpace_scalar(cdg.simulation.simulations.SimulationFeature,
                                  cdg.simulation.simulations.SimulationGreater):

    @classmethod
    def sequence_embedding(cls, embedding_space, dataset,
                           sequence_train, sequence_test, no_train_for_embedding=0,
                           message=None):
        """
        Learns the embedding map based on prototypes. Then it maps the training
        and test grapphs into the embedding space.
        :param embedding_space: space onto which the embedding function maps the
            data. This has to be a subclass instance of
            cdg.embedding.embedding.DissimilarityRepresentation.
        :param dataset: dataset
        :param sequence_train: training sequence
        :param sequence_test: testing sequence
        :param no_train_for_embedding: number of training datapoints in the
            `sequence_train` used to learn the mapping
        :param message: logging string
        :return:
            - x_train : training sequence of embedding points
            - x_test : test sequence of embedding points
        """
        if message is None: message = []

        if not issubclass(type(embedding_space), cdg.embedding.feature.DistanceGraphMean):
            raise cdg.util.errors.CDGForbidden("not very elegant, but works")

        sequence_trainEmbedding = sequence_train[:no_train_for_embedding]

        # select the prototypes
        diss_matrix_prot_sel = dataset.get_sub_dissimilarity_matrix(sequence_trainEmbedding, sequence_trainEmbedding)
        embedding_space.reset()
        embedding_space.fit(dissimilarity_matrix = diss_matrix_prot_sel)

        # dissimilarity representation
        x_train = dataset.get_sub_dissimilarity_matrix(embedding_space.mean_id, sequence_train[no_train_for_embedding:])
        x_test  = dataset.get_sub_dissimilarity_matrix(embedding_space.mean_id, sequence_test)

        return x_train.transpose(), x_test.transpose(), message


class SimulationManifold_scalar(cdg.simulation.simulations.SimulationPrototypeBased,
                                cdg.simulation.simulations.SimulationGreater):

    @classmethod
    def sequence_embedding(cls, embedding_space, dataset,
                           sequence_train, sequence_test, no_train_for_embedding=0,
                           message=None):
        """
        Considers as actual embedding the (scalar) distance between g_t and the mean on the manifold.
        """
        if message is None: message = []
        x_train, x_test, message = super().sequence_embedding(embedding_space=embedding_space,
                                   dataset=dataset,
                                   sequence_train=sequence_train, sequence_test=sequence_test,
                                   no_train_for_embedding=no_train_for_embedding,
                                   message=None)

        x_mean_0 = embedding_space.sample_mean(x_train, radius=embedding_space.radius)
        dist_train = embedding_space.distance(X1=x_mean_0, X2=x_train, radius=embedding_space.radius)
        dist_test  = embedding_space.distance(X1=x_mean_0, X2=x_test,  radius=embedding_space.radius)

        return dist_train.transpose(), dist_test.transpose(), message

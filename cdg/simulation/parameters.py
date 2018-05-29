# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Implements the sets of parameters required for the change detection
# simulations. 
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 10/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import datetime
import numpy as np
import cdg.util.freeze
import cdg.embedding.embedding


class ParametersChangeDetection(cdg.util.freeze.Parameters):
    def set_default(self):
        super().set_default()

        self.creation_time = datetime.datetime.now()
        self.cdg_commit_hash = None

        # Problem [cdg.graph.database]
        self.class0 = None
        self.class1 = None

        # Sequence [cdg.stream.simulation] (lengths, change time and drift)
        self.train_embedding_t = 100
        self.train_changedetection_t = 100

        # Output [cdg.stream.simulation]
        self.name_nominal = 'n.d.'
        self.name_nonnominal = 'n.d.'
        self.launching_command = None
        self.title_tmp = None
        self.filename_tmp = None

        # Embedding [cdg.embedding]
        self.embedding_dimension = 3

        # Method [cdg.changedetection]
        self.significance_level = 0.05
        self.no_simulations_thresh_est = 10000
        # self.lenSimulationThreshEstimation = 100

    def train_total_t(self):
        return self.train_embedding_t + self.train_changedetection_t

    def all_classes(self):
        return self.class0 + self.class1

    def __str__(self):
        string = ''
        for key in self.__dict__.keys():
            string += "{} = {}, ".format(key, self.__dict__[key])
        return string


class ParametersOnline(ParametersChangeDetection):
    """Parameters for online monitring."""

    def set_default(self):
        super().set_default()

        # Sequence [cdg.simulation]
        self.train_embedding_t = 300
        self.train_changedetection_t = 1000
        self.test_nominal_t = 200
        self.test_drift_t = 0
        self.test_nonnominal_t = 200

        # Embedding [cdg.embedding]
        self.manifold = cdg.embedding.embedding.DissimilarityRepresentation().set_parameters(M=3)
        # self.embeddingDimension = 2
        # self.noPrototypes = 3
        self.window_size = 15

        # Method [cdg.changedetection]
        self.beta = .75
        self.significance_level = 0.01

    def arl_w(self):
        return np.ceil(1. / self.significance_level)

    def arl_t(self):
        return np.round(self.arl_w() * self.window_size)

    def test_total_t(self):
        return self.test_nominal_t + self.test_drift_t + self.test_nonnominal_t

    def ratio_nominal(self):
        return self.test_nominal_t / self.test_total_t()

    def ratio_nonnominal(self):
        return self.test_nonnominal_t / self.test_total_t()

    def test_total_w(self):
        return np.ceil(self.test_total_t() // self.window_size)

    def train_total_w(self):
        return np.ceil(self.train_total_t() // self.window_size)

    def set_drift_change(self, prc=0.5):
        self.test_drift_t = int(self.test_nominal_t * prc)
        self.test_nominal_t -= self.test_drift_t


class ParametersOffline(ParametersChangeDetection):
    """Parameters for online monitring."""

    def set_default(self):
        super().set_default()

        # Sequence [cdg.simulation]
        self.train_embedding_t = 300
        self.train_changedetection_t = 1000
        self.total_nominal_t = 200
        self.total_nonnominal_t = 200

        # Embedding [cdg.embedding]
        self.manifold = cdg.embedding.embedding.DissimilarityRepresentation().set_parameters(M=3)

        # Method [cdg.changedetection]
        self.significance_level = 0.01

    def total_t(self):
        return self.total_nominal_t + self.total_nonnominal_t

    def ratio_nominal(self):
        return self.total_nominal_t / self.total_total_t()

    def ratio_nonnominal(self):
        return self.total_nonnominal_t / self.total_total_t()

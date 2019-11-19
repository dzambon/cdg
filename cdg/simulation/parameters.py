# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements the sets of parameters required for simulating repeated experiments.
# --------------------------------------------------------------------------------
import datetime
import numpy as np
import cdg.utils
import cdg.utils.freeze
import cdg.embedding.embedding


class Parameters(cdg.utils.freeze.Freezable, cdg.utils.Loggable, cdg.utils.Pickable):
    """
    Generic class for parameters.
    """
    
    description = {}

    def __init__(self):
        cdg.utils.freeze.Freezable.__init__(self)
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        self.addtowhitelist('_log')
        self._set_default_values()
        self.close()

    def __setattr__(self, key, value):
        try:
            cdg.utils.freeze.Freezable.__setattr__(self, key, value)
            self.log.info("setting parameter %s: %s" % (key, value))
        except KeyError as e:
            raise cdg.utils.ForbiddenError(str(e))

    def _define_parameter(self, name, value, description='...'):
        setattr(self, name, value)
        self.description[name] = description
        
    def _set_default_values(self):
        self.log.info(' *** default parameters of class Parameters')
        # Output [cdg.stream.simulation]
        # self.creation_time = datetime.datetime.now()
        self._define_parameter('creation_time', datetime.datetime.now(), '(automatic) time-stamp of the creation time.')
        # self.launching_command = None
        self._define_parameter('launching_command', None, '(automatic) command with which the simulation was runned.')
        # self.title_tmp = None
        # self.filename_tmp = None
        # self.cdg_commit_hash = None
        self._define_parameter('cdg_commit_hash', None, 'version of cdg that was used.')

    def info(self):
        desc_str = ''
        for par in self.description:
            desc_str += '{:<30}:\t{}\n'.format(par, self.description[par])
        return desc_str
        
        
class ParametersChangeDetection(Parameters):

    def _set_default_values(self):
        super()._set_default_values()
        self.log.info(' *** default parameters of class ParametersChangeDetection')
        self.creation_time = datetime.datetime.now()

        # Sequence [cdg.simulation]
        # self.sequence_len_t = None
        # self._define_parameter('sequence_len_t', None, 'length of the entire sequence in time-steps.')

        # Embedding [cdg.embedding]
        # self.embedding_dimension = 3
        self._define_parameter('embedding_dimension', 3, 'dimension of the embedding space')
        # self.embedding_method = cdg.embedding.DissimilarityRepresentation(emb_dim=3)
        self._define_parameter('embedding_method',
                               cdg.embedding.DissimilarityRepresentation(emb_dim=self.embedding_dimension),
                               'instance of class cdg.embedding.Embedding')
        self._define_parameter('no_prototypes', 3, 'dimension of the embedding space')

        # Method [cdg.changedetection]
        # self.significance_level = 0.05
        self._define_parameter('significance_level', 0.05, 'significance level of the test: alpha = P(reject H0 | H0 is true).')

    # def train_total_t(self):
    #     return self.train_embedding_t + self.train_changedetection_t

    def all_classes(self):
        return self.class0 + self.class1

    def __str__(self):
        string = ''
        for key in self.__dict__.keys():
            string += "{} = {}, ".format(key, self.__dict__[key])
        return string


class ParametersCDT(ParametersChangeDetection):
    """Parameters for online monitring."""
    def __init__(self, **kwargs):
        raise NotImplementedError() # todo


class ParametersCPM(ParametersChangeDetection):
    """Parameters for offline monitring."""

    def _set_default_values(self):
        super()._set_default_values()
        self.log.info(' *** default parameters of class ParametersCPM')

        # Problem [cdg.graph.dataset]
        # self.classes = None
        self._define_parameter('classes', None, 'list of class labels that will define regimes in between change points.')
        self._define_parameter('subseq_lengths_t', None, 'list of lengths of each of the sub-sequences associated with the list of classes.')
        self._define_parameter('subseq_ratios', None, 'list of ratios of graphs extracted from the corresponding class.')

        # Sequence [cdg.simulation]
        # self.train_embedding_ratio = .5
        self._define_parameter('train_len_ratio', 0.5, 'proportion on the input data employed to train the embedding.')
        # self.train_changedetection_t = 1000
        # self.total_nominal_t = 200
        # self.total_nonnominal_t = 200
        # self.margin = 15
        self._define_parameter('margin', 15, 'the two-sample test statistics are computed only in time steps [margin: -margin].')

        # Method [cdg.changedetection]
        # self.cpm = cdg.changedetection.cpm.EDivisive_R(R=199)
        # self._define_parameter('cpm',
        #                        cdg.changedetection.cpm.EnergyCPM(),
        #                        'Change-point method: instance of class cdg.changedetection.cpm.')

    @property
    def sequence_len_t(self):
        return None if self.subseq_lengths_t is None else sum(self.subseq_lengths_t)

    @property
    def train_len_t(self):
        return int(self.train_len_ratio * self.sequence_len_t)
    
    @train_len_t.setter
    def train_len_t(self, value):
        self.train_len_ratio = 1. * value / self.sequence_len_t

    @property
    def test_len_t(self):
        return self.sequence_len_t - self.train_len_t

    # def total_t(self):
    #     return self.total_nominal_t + self.total_nonnominal_t
    #
    # def ratio_nominal(self):
    #     return self.total_nominal_t / self.total_total_t()
    #
    # def ratio_nonnominal(self):
    #     return self.total_nonnominal_t / self.total_total_t()

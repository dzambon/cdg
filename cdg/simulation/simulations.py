# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Repeated experiments of change detection on a graph stream.
#
#
# References:
# ---------
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
# ------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# ------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 25/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import os
import subprocess
import shutil
import numpy as np
import scipy
import scipy.stats
import cdg.util.errors
import cdg.changedetection.cusum
import cdg.graph
import cdg.embedding.manifold
import cdg.util.prototype
import cdg.util.logger
import cdg.util.serialise


def binom_se(p, n):
    return np.sqrt(p * (1 - p) / n)


def binom_ci95(p, sim):
    # ISSUE: for p=0 scipy.binom.ppf return n-1
    # https://github.com/scipy/scipy/issues/1603
    # https://github.com/scipy/scipy/issues/5122
    # This code is work around to by pass it
    if p == 0:
        a = 0.0
        b = 0.0
    elif p == 1:
        a = 1.0
        b = 1.0
    else:
        a = scipy.stats.binom.ppf(q=.025, n=sim, p=p, loc=0) / sim
        b = scipy.stats.binom.ppf(q=.975, n=sim, p=p, loc=0) / sim
    return a, b


def process_run_lengths(runLen, default=-1):
    meanRunLen = []
    alarms = []
    cleanMeanRunLen = []
    for rl in runLen:
        alarms.append(len(rl))
        if len(rl) > 0:
            meanRunLen.append(np.mean(rl))
            cleanMeanRunLen.append(np.mean(rl))
        else:
            meanRunLen.append(default)
    return meanRunLen, alarms, cleanMeanRunLen


class Simulation(cdg.util.logger.Loggable, cdg.util.serialise.Pickable):
    """
    Parent class dealing with the parameters and performance assessment.
    """

    # this is a list of known exception that are known that may occur, and
    # which we want to deal with
    _controlled_exceptions = (np.linalg.LinAlgError)

    def __init__(self):
        cdg.util.logger.Loggable.__init__(self)
        # cdg.util.serialise.Pickable.__init__(self)
        self.skip_from_serialising(['dataset'])

        self.pars = None
        self.dataset = None
        self.no_simulations = None

    def set(self, parameters, dataset, no_simulations, folder,
            load_dataset=True):
        """

        :param parameters: instance of cdg.stream.parameters.Parameters
        :param dataset: instance of cdg.graph.database.Database
        :param no_simulations: number of repeated experiments in the same
            parameter setting
        :param folder: absolute path to folder in which to store the outcome
        :param load_dataset: flag for whether to load precomputed
            dissimilarities
        """
        self.pars = parameters
        self.dataset = dataset
        self.no_simulations = no_simulations
        self.folder = folder

        # load dataset
        if load_dataset:
            self.log.info('retrieving precomputed dissimilarities...')
            self.dataset.load_graph_name_map()
            self.dataset.load_dissimilarity_matrix()

    def empty_results(self):
        pass

    def run(self, seed=None, logfile=None):
        """
        Run repeated simulations and perform some processing of the results.
        :param seed: (None) global seed for replicability
        :return: run lengths under H0, run lengths under H1
        """

        # empty results
        self.empty_results()

        # output: failed simulations
        max_num_failure = self.no_simulations
        self.fails = []

        # set global seed
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

        # create output folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # auxiliar variables
        h_threshold = None
        ctSimulation = 0

        while ctSimulation < self.no_simulations:

            self.log.info("Running simulation {} / {}"
                          .format(ctSimulation + 1, self.no_simulations))

            try:
                # run simulation
                h_threshold = self.run_single_simulation(threshold=h_threshold)
                # count simulation as correctly terminated
                ctSimulation += 1
                self.log.debug("status = completed")

            except self._controlled_exceptions as e:
                # here goas a list o
                # record failed simulation
                self.fails.append(ctSimulation)
                self.log.warning("status = failed: {}".format(e))

            if len(self.fails) > max_num_failure:
                raise cdg.util.errors.CDGError(
                    'We reached %d failed simulation.' % len(self.fails))

        self.serialise(self.folder + "/simulation.pkl")
        self.write_results()

        # Compress results in a .zip archive
        if logfile is not None:
            shutil.move(logfile, self.folder + '/' + logfile)
        command = 'zip -m ' + self.folder + '.zip ' + self.folder + ' -r '
        self.log.debug("executing: " + command)
        subprocess.Popen(command.split()).wait()

    def run_single_simulation(self, threshold=None):
        raise cdg.util.errors.CDGAbstractMethod()

    def string_raw_results(self):
        raise cdg.util.errors.CDGAbstractMethod()

    def process_results(self, figures_merit=None):
        raise cdg.util.errors.CDGAbstractMethod()

    def write_results(self, figures_merit=None):
        """
        This function receives a list a figures of merit, compute them and
        print them out to a predefined file.

        :param figureMerit: list of figure of merit to be computed
        :return:
        """

        # open output file
        f = open(self.folder + "/000_experiment_setup_and_results.txt", 'w')

        # parameters and settings of the simulation
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Setting           # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines(self.folder + '\n')
        f.writelines(str(self.pars))
        f.writelines(str(self.dataset))
        f.writelines('\n')
        f.writelines(
            'number of concluded simulation:' + str(self.no_simulations) + '\n')
        f.writelines('seed:' + str(self.seed) + '\n')
        f.writelines('\n\n')

        # raw results
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Raw results       # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines(self.string_raw_results() + '\n')
        f.writelines('failed simulations:' + str(self.fails) + '\n')
        f.writelines('\n\n')

        # processed rusults
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Processed results # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        tabularResult, printout = self.process_results(figures_merit=figures_merit)
        f.writelines(printout + '\n')

        # close file
        f.close()

        self.log.info(tabularResult)
        return tabularResult

    @staticmethod
    def sequence_generator(dataset, pars):
        raise cdg.util.errors.CDGAbstractMethod()


class SimulationOnline(Simulation):
    def empty_results(self):
        self.runLen0 = []
        self.runLen1 = []

    def run_single_simulation(self, threshold=None):
        """
        Implements the specific simulation of this class
        :param threshold: (None) a predefined threshold for the cusum, otherwise estimated
        :return: adopted_threshold
        """

        self.log.debug('embedding...')
        x_train, x, message = self.generate_embedding_part(dataset=self.dataset,
                                                           pars=self.pars)

        [self.log.debug(m) for m in message]

        self.log.debug('training...')
        cusum = self.training_part(x_train, self.pars, threshold=threshold)

        self.log.debug('operating phase...')
        self.logplot(
            [x[:self.pars.test_nominal_t, :2], x[self.pars.test_nominal_t:, :2]],
            stylelist=['xg', '+r'])

        runLen0_tmp, runLen1_tmp, charplot, dd = \
            self.operating_phase(self.pars, cusum, x)
        self.log.info(charplot)
        self.logplot([np.array(dd)], stylelist=['-'])

        # store estimated times
        self.runLen0.append(runLen0_tmp)
        self.runLen1.append(runLen1_tmp)

        return cusum.threshold

    @classmethod
    def generate_embedding_part(cls, dataset, pars, message=None):
        """

        :param dataset: considered dataset
        :param pars: parameters
        :param message: possible message for logging
        :return:
            - x_train : sequence to be used as training set
            - x : sequence to be used as testing sequence
            - message : logging message
        """
        if message is None:
            message = []

        streamTrain, streamTest = cls.sequence_generator(dataset=dataset,
                                                         pars=pars)
        x_train, x, message = cls.sequence_embedding(dataset=dataset,
                                                     embedding_space=pars.manifold,
                                                     sequence_train=streamTrain,
                                                     sequence_test=streamTest,
                                                     no_train_for_embedding=pars.train_embedding_t)
        return x_train, x, message

    @staticmethod
    def sequence_generator(dataset, pars):
        """
        Bootstrap the graphs, select the prototypes and compute all the
        dissimilarities.

        :param dataset: dataset
        :param pars: parameters
        :return:
           - sequence_training : sequence used as training set
           - sequence_test : sequence (stream) to be monitored

        """

        # bootstrap
        sequence_training = \
            dataset.generate_bootstrapped_stream(pars.class0,
                                                 pars.train_total_t())
        sequence_test_0 = \
            dataset.generate_bootstrapped_stream(pars.class0,
                                                 pars.test_nominal_t)
        sequence_test_1 = \
            dataset.generate_bootstrapped_stream(pars.class1,
                                                 pars.test_nonnominal_t)

        # set up smooth drift
        if pars.test_drift_t > 0:
            raise NotImplementedError()
        else:
            sequence_test_d = []

        # assemble stream
        sequence_test = sequence_test_0 + sequence_test_d + sequence_test_1

        return sequence_training, sequence_test

    @classmethod
    def training_part(cls, y_train, pars, threshold):
        raise cdg.util.errors.CDGAbstractMethod()

    @staticmethod
    def operating_phase(pars, cusum, x):
        """
        Launch the cusum on the windowed data
        :param cusum: cusum instance
        :param y: all test data, numPrototyeps \times numTestWindow
        :return:
        """

        # Auxiliar variables
        # variables for run lengths
        runLen0 = []
        runLen1 = []
        # vars for change time
        ctWin = 0  # window unit
        ctTime = 0  # time unit
        alarmWin = -1
        alarmTime = -1
        changeTime = int(np.floor(pars.test_nominal_t)) - 1
        changeWin = changeTime // pars.window_size
        changeOccurred = False
        statusBarLen = 70
        noTotalWin = int(np.floor(pars.test_nominal_t) + np.floor(
            pars.test_nonnominal_t)) // pars.window_size
        statusBarStep = int(np.floor(1. * noTotalWin / statusBarLen))

        # init cusum
        cusum.reset()
        alarm = False
        alarmStatusBar = False

        dd = []
        # loop
        if x.shape[1] != pars.test_total_t():
            cdg.util.errors.CDGError("the length of the data is not correct.")
        charplot = ''
        while ctTime < pars.test_total_t():

            # status bar
            alarmStatusBar = alarm or alarmStatusBar
            if ctWin == changeWin:
                changeOccurred = True
                sys.stdout.write('|')
                charplot += '|'
            if ctWin % statusBarStep == 0:
                if alarmStatusBar:
                    sys.stdout.write("'")
                    charplot += "'"
                else:
                    sys.stdout.write('.')
                    charplot += '.'
                alarmStatusBar = False

            # Iterate
            alarm, inc = cusum.iterate(x[ctTime: ctTime + pars.window_size, :],
                                       reset=False)
            dd.append(inc)

            # Checks alarms
            if alarm:

                if not changeOccurred:
                    runLen0.append(cusum.time)

                else:
                    if len(runLen1) == 0:
                        alarmWin = ctWin
                        alarmTime = ctTime
                        runLen1.append(ctWin - changeWin)
                    else:
                        runLen1.append(cusum.time)

                cusum.reset()

            # Time step update
            ctWin += 1
            ctTime += pars.window_size

        sys.stdout.write('\n')

        return runLen0, runLen1, charplot, dd

    def string_raw_results(self):
        return 'runLen0 = ' + str(self.runLen0) + '\n' + 'runLen1 = ' + str(
            self.runLen1)

    def process_results(self, figures_merit):
        """
        Process the outcomes for synthetic results.
        :param figures_merit: list of figures of merit to be printed
        """

        if figures_merit is None:
            figures_merit = ['dca_rate']

        printout = ''
        result = {}

        # process alarms
        try:
            meanRunLen0, falseAlarms, cleanMeanRunLen0 = process_run_lengths(
                self.runLen0, None)
            meanRunLen1, trueAlarms, cleanMeanRunLen1 = process_run_lengths(
                self.runLen1, None)
            # meanRunLen1, trueAlarms, _  = processAlarms(runLen1, -1) 
        except ValueError as e:
            self.log.error(
                "so sad... probably runLen0 or runLen1 is empty: " + e)

        if len(meanRunLen0) != len(meanRunLen1) or len(
                meanRunLen0) != self.no_simulations:
            raise cdg.util.errors.CDGImpossible('len(meanRunLen0) != len(meanRunLen1)...')

        try:
            # Detected Change Rate (DCR) (equivalently : Test hp: observed ARL1 = target ARL0)
            try:
                dc = 0
                for s in range(0, self.no_simulations):
                    if trueAlarms[s] == 0:
                        dc += 0  # do nothing
                    elif meanRunLen1[s] < self.pars.arl_w():
                        dc += 1
                dc_rate = 1. * dc / self.no_simulations
                dc_rate_std = binom_se(dc_rate, self.no_simulations)
                dc_rate_a, dc_rate_b = binom_ci95(dc_rate, self.no_simulations)
            except ValueError as e:
                self.log.error("so sad... probably trueAlarms is empty: " + e)
                dc_rate = dc_rate_std = dc_rate_a = dc_rate_b = -1
            printout += 'detected changes rate :  DCR (std) [95 conf.int.] = %.3f (%.3f) [%.3f, %.3f]\n' % (
                dc_rate, dc_rate_std, dc_rate_a, dc_rate_b)
            result['dc_rate'] = '%.3f' % dc_rate
            result['dc_rate_std'] = '%.3f' % dc_rate_std
            result['dc_rate_a'] = '%.3f' % dc_rate_a
            result['dc_rate_b'] = '%.3f' % dc_rate_b
            result['dc_rate_95ci'] = '[%s, %s]' % (
                result['dc_rate_a'], result['dc_rate_b'])

            # Test hp: observed ARL0 = target ARL0
            try:
                rl_bi = []
                for s in range(0, self.no_simulations):
                    if falseAlarms[s] == 0:
                        rl_bi.append(0)
                    elif meanRunLen0[s] > self.pars.arl_w():
                        rl_bi.append(0)
                    else:
                        rl_bi.append(1)

                rl0_bi_p = sum(rl_bi) * 1. / self.no_simulations
                rl0_bi_a, rl0_bi_b = binom_ci95(rl0_bi_p, self.no_simulations)
            except ValueError as e:
                self.log.warning("penso che runLen0 sia vuoto... Error: ")
                self.log.warning(e)
                rl0_bi_p = rl0_bi_a = rl0_bi_b = -1
            printout += 'test per ARL0 : mean [95 c.i.] = %.3f [%.0f, %.0f]\n' % (
                rl0_bi_p, rl0_bi_a, rl0_bi_b)
            result['rl0_bi_p'] = '%.3f' % rl0_bi_p
            result['rl0_bi_a'] = '%.3f' % rl0_bi_a
            result['rl0_bi_b'] = '%.3f' % rl0_bi_b
            result['arl0_bi_95ci'] = '[%s, %s]' % (
                result['rl0_bi_a'], result['rl0_bi_b'])

            # Detected Change Rate adapted (DCRa) (equivalently : Test hp: observed ARL1 = observed ARL0)
            try:
                dca = 0
                for s in range(0, self.no_simulations):
                    if trueAlarms[s] == 0:
                        dca += 0  # do nothing
                    elif falseAlarms[s] == 0:
                        dca += 1
                    elif meanRunLen1[s] < meanRunLen0[s]:
                        dca += 1
                dca_rate = 1. * dca / self.no_simulations
                dca_rate_std = binom_se(dca_rate, self.no_simulations)
                dca_rate_a, dca_rate_b = binom_ci95(dca_rate,
                                                    self.no_simulations)
            except ValueError as e:
                self.log.warning("penso che meanRunLen1 sia vuoto... Error: ")
                self.log.warning(e)
                dca_rate = dca_rate_std = dca_rate_a = dca_rate_b = -1
            printout += 'detected changes rate adapted:  DCRa (std) [95 conf.int.] = %.3f (%.3f) [%.3f, %.3f]\n' % (
                dca_rate, dca_rate_std, dca_rate_a, dca_rate_b)
            result['dca_rate'] = '%.3f' % dca_rate
            result['dca_rate_std'] = '%.3f' % dca_rate_std
            result['dca_rate_a'] = '%.3f' % dca_rate_a
            result['dca_rate_b'] = '%.3f' % dca_rate_b
            result['dca_rate_95ci'] = '[%s, %s]' % (
                result['dca_rate_a'], result['dca_rate_b'])

            # Average Run Length 0
            try:
                arl0 = np.mean(cleanMeanRunLen0)
                rl0_00 = min(cleanMeanRunLen0)
                rl0_025 = scipy.percentile(cleanMeanRunLen0, 2.5)
                rl0_05 = scipy.percentile(cleanMeanRunLen0, 5)
                rl0_25 = scipy.percentile(cleanMeanRunLen0, 25)
                rl0_50 = scipy.percentile(cleanMeanRunLen0, 50)
                rl0_75 = scipy.percentile(cleanMeanRunLen0, 75)
                rl0_95 = scipy.percentile(cleanMeanRunLen0, 95)
                rl0_975 = scipy.percentile(cleanMeanRunLen0, 97.5)
                rl0_100 = max(cleanMeanRunLen0)
            except ValueError as e:
                self.log.warning("penso che runLen1 sia vuoto... Error: ")
                self.log.warning(e)
                arl0 = rl0_00 = rl0_025 = rl0_05 = rl0_25 = rl0_50 = rl0_75 = rl0_95 = rl0_975 = rl0_100 = -1
            printout += 'average run length in the nominal regime : ARL0 [IQ interval] = %.0f [%.0f, %.0f]\n' % (
                arl0, rl0_25, rl0_75)
            printout += 'average run length in the nominal regime : [0.00,.05,.25,.50,.75,.95,1.00] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n' % (
                rl0_00, rl0_05, rl0_25, rl0_50, rl0_75, rl0_95, rl0_100)
            result['arl0'] = '%.0f' % arl0
            result['rl0_00'] = '%.0f' % rl0_00
            result['rl0_025'] = '%.0f' % rl0_025
            result['rl0_05'] = '%.0f' % rl0_05
            result['rl0_25'] = '%.0f' % rl0_25
            result['rl0_50'] = '%.0f' % rl0_50
            result['rl0_75'] = '%.0f' % rl0_75
            result['rl0_95'] = '%.0f' % rl0_95
            result['rl0_975'] = '%.0f' % rl0_975
            result['rl0_100'] = '%.0f' % rl0_100
            result['arl0_95ci'] = '[%s, %s]' % (
                result['rl0_025'], result['rl0_975'])

            # Average Run Length 1  (Delay of Detection)
            try:
                arl1 = np.mean(cleanMeanRunLen1)
                rl1_00 = min(cleanMeanRunLen1)
                rl1_025 = scipy.percentile(cleanMeanRunLen1, 2.5)
                rl1_05 = scipy.percentile(cleanMeanRunLen1, 5)
                rl1_25 = scipy.percentile(cleanMeanRunLen1, 25)
                rl1_50 = scipy.percentile(cleanMeanRunLen1, 50)
                rl1_75 = scipy.percentile(cleanMeanRunLen1, 75)
                rl1_95 = scipy.percentile(cleanMeanRunLen1, 95)
                rl1_975 = scipy.percentile(cleanMeanRunLen1, 97.5)
                rl1_100 = max(cleanMeanRunLen1)
            except ValueError as e:
                self.log.warning("penso che runLen1 sia vuoto... Error: ")
                self.log.warning(e)
                arl1 = rl1_00 = rl1_025 = rl1_05 = rl1_25 = rl1_50 = rl1_75 = rl1_95 = rl1_975 = rl1_100 = -1
            printout += 'average run length in the out-of-control regime : DoD [IQ interval] = %.0f [%.0f, %.0f]\n' % (
                arl1, rl1_25, rl1_75)
            printout += 'average run length in the out-of-control regime : [0.00,.05,.25,.50,.75,.95,1.00] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n' % (
                rl1_00, rl1_05, rl1_25, rl1_50, rl1_75, rl1_95, rl1_100)
            result['arl1'] = '%.0f' % arl1
            result['rl1_00'] = '%.0f' % rl1_00
            result['rl1_025'] = '%.0f' % rl1_025
            result['rl1_05'] = '%.0f' % rl1_05
            result['rl1_25'] = '%.0f' % rl1_25
            result['rl1_50'] = '%.0f' % rl1_50
            result['rl1_75'] = '%.0f' % rl1_75
            result['rl1_95'] = '%.0f' % rl1_95
            result['rl1_975'] = '%.0f' % rl1_975
            result['rl1_100'] = '%.0f' % rl1_100
            result['arl1_95ci'] = '[%s, %s]' % (
                result['rl1_025'], result['rl1_975'])

            # False alarms within 1000 samples
            try:
                scaleFactor = 1000 / (self.pars.test_nominal_t)
                fa1000_rate = np.mean(falseAlarms) * scaleFactor
                fa1000_rate_std = np.std(falseAlarms) * scaleFactor
                fa1000_rate_25 = scipy.percentile(falseAlarms, 25) * scaleFactor
                fa1000_rate_75 = scipy.percentile(falseAlarms, 75) * scaleFactor
            except ValueError as e:
                self.log.warning("penso che runLen1 sia vuoto... Error: ")
                self.log.warning(e)
                fa1000_rate = fa1000_rate_std = fa1000_rate_25 = fa1000_rate_75 = -1
            printout += 'false alarms rate per 1000 sample : FA1000 (std) [IQ interval] = %.3f (%.3f) [%.0f, %.0f]\n' % (
                fa1000_rate, fa1000_rate_std, fa1000_rate_25, fa1000_rate_75)
            result['fa1000_rate'] = '%.3f' % fa1000_rate
            result['fa1000_rate_std'] = '%.3f' % fa1000_rate_std
            # result['fa1000_rate_25'] = fa1000_rate_25
            # result['fa1000_rate_75'] = fa1000_rate_75

            # # Latex table entries

            printout += '\n\n'
            printout += '# # # # # # # # # # # # # # #' + '\n'
            printout += '# # # Latex table entry # # #' + '\n'
            printout += '# # # # # # # # # # # # # # #' + '\n'

            closingString = ' \\\\'
            try:
                closingString += '\t % %s\n' % self.dataset.notes
            except:
                closingString += '\n'

            if figures_merit[0] == 'matlab':
                figures_merit = [
                    '{[dca rate ,  dca rate a ,  dca rate b , no Prot ,  win Size] ,  "M..n.._name"}']
                selectedResults = ["{[%f,%f,%f], 'M=%d, n=%s'}, ... %s" % (
                    dca_rate, dca_rate_a, dca_rate_b, self.pars.embeddingDimension,
                    self.pars.window_size, self.dataset)]
            else:
                selectedResults = []
                for figMer in figures_merit:
                    selectedResults.append(result[figMer])

            # # Footer message
            printout += '\n\nAll went well, apparently.\n\n'
            return [figures_merit, selectedResults], printout

        except IndexError:
            self.log.warning('something went wrong in computing the results.')
            printout += '....something went wrong....'
            return None, printout


class SimulationPrototypeBased(SimulationOnline):
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

        if not issubclass(type(embedding_space),
                          cdg.embedding.embedding.DissimilarityRepresentation):
            raise cdg.util.errors.CDGForbidden("not very elegant, but works")

        sequence_trainEmbedding = sequence_train[:no_train_for_embedding]

        # select the prototypes
        diss_matrix_prot_sel = dataset.get_sub_dissimilarity_matrix(sequence_trainEmbedding,
                                                                    sequence_trainEmbedding)
        if not np.allclose(diss_matrix_prot_sel, diss_matrix_prot_sel.transpose()):
            msg = 'The dissimilarity matrix is not symmetric; I will make it symmetric'
            cdg.util.logger.glog().warning(msg)
            message.append(msg)
            diss_matrix_prot_sel += diss_matrix_prot_sel.transpose()
            diss_matrix_prot_sel *= .5

        embedding_space.reset()
        embedding_space.fit(dissimilarity_matrix=diss_matrix_prot_sel, no_annealing=20)
        prototypes = [sequence_trainEmbedding[i] for i in embedding_space.prototype_indices]

        # dissimilarity representation
        y_train = dataset.get_sub_dissimilarity_matrix(prototypes,
                                                       sequence_train[no_train_for_embedding:])
        y_test = dataset.get_sub_dissimilarity_matrix(prototypes, sequence_test)

        # manifold representation
        cdg.util.logger.glog().debug("x_train starting...")
        x_train = embedding_space.predict(y_train.transpose())
        x_test = embedding_space.predict(y_test.transpose())

        return x_train, x_test, message


class SimulationFeature(SimulationOnline):
    """
    Simulation monitoring a single feature.
    """

    @classmethod
    def sequence_embedding(cls, embedding_space, dataset,
                           sequence_train, sequence_test, no_train_for_embedding=0,
                           message=None):

        if message is None: message = []

        if not issubclass(type(embedding_space),
                          cdg.embedding.feature.GraphFeature):
            raise cdg.util.errors.CDGForbidden("not very elegant, but works")

        x_train = embedding_space.predict(graph_list=sequence_train, dataset=dataset)
        x_test = embedding_space.predict(graph_list=sequence_test, dataset=dataset)

        return x_train, x_test, message


class SimulationCLT(SimulationOnline):
    """ Adopts the cdg.changedetection.GaussianCusum change detection test (CDT). """

    @classmethod
    def training_part(cls, x_train, pars, threshold=None):
        """
        :param x_train: sequence used to train the CDT
        :param pars:parameters
        :param threshold: threshold for the CDT
        :return:
        """
        cusum = cdg.changedetection.cusum.GaussianCusum(arl=pars.arl_w(),
                                                        window_size=pars.window_size)
        if threshold is None:
            cusum.fit(x=x_train, beta=pars.beta, estimate_threshold=True,
                      gamma_type='quantile',
                      len_simulation=pars.no_simulations_thresh_est)
        else:
            cusum.fit(x=x_train, beta=pars.beta, estimate_threshold=False,
                      gamma_type='quantile')
            cusum.threshold = threshold
        return cusum


class SimulationDifference(SimulationOnline):
    """ 
    Adopts one of the the cdg.changedetection.cusum.DifferenceCusum as change detection test (CDT).
    The specific method is selected in one of its subclasses.
    """
    _cusum_difference = cdg.changedetection.cusum.DifferenceCusum

    @classmethod
    def training_part(cls, x_train, pars, threshold=None):
        # Train the cusum
        if pars.window_size != 1:
            raise cdg.util.errors.CDGError(
                "window size must be 1, but %d was given" % pars.window_size)
        cusum = cls._cusum_difference(arl=pars.arl_w())
        cusum.fit(x=x_train, beta=pars.beta, estimate_threshold=True)
        return cusum


class SimulationGreater(SimulationDifference):
    _cusum_difference = cdg.changedetection.cusum.GreaterCusum


class SimulationLower(SimulationDifference):
    _cusum_difference = cdg.changedetection.cusum.LowerCusum


class SimulationTwoSided(SimulationDifference):
    _cusum_difference = cdg.changedetection.cusum.TwoSidedCusum

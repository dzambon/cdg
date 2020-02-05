# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Implements functionalities for repeated experiments.
# --------------------------------------------------------------------------------
import sys
import os
import subprocess
import shutil
from collections import OrderedDict
import numpy as np
import scipy
import scipy.stats
import cdg.utils
import sklearn.metrics

import cdg.changedetection
import cdg.graph
import cdg.embedding

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

def create_foldername(prefix, exp_name, pars, cpm, main_folder=None):
    '''A predefined format for naming the folder that will contain the results.'''
    cls_str = ''
    for c in pars.classes:
        cls_str += '-'+str(c)
    folder_format = '{prefix}{time}[{exp_name}]emb_{emb}_cls_{cls}_cpm{cpm}'
    folder_str = folder_format.format(prefix=prefix,
                                      time=pars.creation_time.strftime('%G%m%d_%H%M_%S'),
                                      exp_name=exp_name,
                                      emb=pars.embedding_method,
                                      cls=cls_str,
                                      cpm=cpm.name)
    
    if main_folder is None:
        pass
    elif main_folder[-1] == '/':
        folder_str = '{}{}'.format(main_folder, folder_str)
    else:
        folder_str = '{}/{}'.format(main_folder, folder_str)
    return folder_str

def get_figures_merit():
    '''Collect list (actually a dict) of figures of merit from both the online and offline cases.'''
    figure_merit_online = SimulationCDT.figure_merit_list()
    figure_merit_offline = SimulationCPM.figure_merit_list()

    all_figures_merit = figure_merit_online.copy()
    all_figures_merit.update(figure_merit_offline)
    return  all_figures_merit

def read_resultfile(zipped_file_path, figure_merit, params, simulation_fname='simulation.pkl'):
    import tempfile
    import zipfile
    import shutil

    # read zipped result file
    cdg.utils.logger.info('computing results...')
    assert os.path.isfile(zipped_file_path), 'zip file {} not found. Try with the absolute path.'.format(
        zipped_file_path)
    zfile = zipfile.ZipFile(zipped_file_path)
    
    # path to result file relative to the zipped file
    zipped_file_path_no_extension = zipped_file_path[:-4]
    read_from_here = zipped_file_path_no_extension.rfind('/') + 1
    simulation_file_relpath = '{}/{}'.format(zipped_file_path_no_extension[read_from_here:], simulation_fname)
    
    # unzip result file
    unzip_dir_temp = tempfile.mkdtemp(prefix='cdg_read_resultfile')
    simulation_file_unzipped = zfile.extract(simulation_file_relpath, path=unzip_dir_temp)
    
    # deserialise simulation
    simulation = Simulation.deserialise(simulation_file_unzipped)
    
    # remove temporary folder
    shutil.rmtree(unzip_dir_temp, ignore_errors=True)
    
    # process result file
    fig_mer_list, results, _ = simulation.process_results(figure_merit)
    # fig_mer_list, results = tmp[0], tmp[1]
    
    selected_results = OrderedDict()
    # for fig_mer in results.keys():
    #     if fig_mer in fig_mer_list:
    #         selected_results[fig_mer] = results[fig_mer]
    for fig_mer in fig_mer_list:
        selected_results[fig_mer] = results[fig_mer]

    settings = OrderedDict()
    for p in params:
        if p == 'debug':
            settings[p] = 'debug'
        elif p == 'cpm':
            settings[p] = None if simulation.cpm is None else simulation.cpm.name
        elif p == 'emb_method':
            settings[p] = simulation.pars.embedding_method.name
        elif p == 'emb_dim':
            settings[p] = simulation.pars.embedding_dimension
        elif p == 'dataset':
            settings[p] = None if simulation.dataset is None else simulation.dataset.name
        elif p == 'classes':
            cls_str = ''
            for c in simulation.pars.classes:
                cls_str += '.{}'.format(str(c))
            settings[p] = cls_str
        else:
            settings[p] = simulation.__getattribute__(p)
        
    return selected_results, settings
    # tabularResult, _ = main(sys.argv[1:])
    # print(tabularResult[1])
    # for i in range(0, len(tabularResult[0][0])):
    #     print(tabularResult[0][0][i] + ':\t' + tabularResult[0][1][i])


class Simulation(cdg.utils.Loggable, cdg.utils.Pickable):
    """ Parent class dealing with the parameters and performance assessment. """

    # this is a list of known exception that are known that may occur, and
    # which we want to deal with
    _controlled_exceptions = (np.linalg.LinAlgError, cdg.utils.EDivisiveRFileNotFoundError)
    _figure_merit_list = {None: ''}
    _exp_result_and_setup_file = "/000_experiment_setup_and_results.txt"
    def __init__(self):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        # self.skip_from_serialising(['dataset'])

        self.pars = None
        self.dataset = None
        self.no_simulations = None

    def set(self, parameters, dataset, no_simulations,
            folder='./cdg_result', load_dataset=True):
        """

        :param parameters: instance of class Parameters
        :param dataset: instance of cdg.graph.database.Database # todo
        :param no_simulations: number of repeated experiments in the same
            parameter setting
        :param folder: absolute path to folder in which to store the outcome
        :param load_dataset: flag for whether to load precomputed
            dissimilarities
        """
        # todo maybe parameters can be passed as a dictionary, and the creation of Parameters instance
        # is performed directly here. There is a need of parameters that user can set
        self.pars = parameters
        self.dataset = dataset
        self.no_simulations = no_simulations
        self.folder = folder

        if load_dataset and not dataset.is_loaded:
            self.log.info('loading dataset...')
            self.dataset.load()

    def empty_results(self):
        pass

    def run(self, seed=None, logfile=None, **kwargs):
        """
        Run repeated simulations and perform some processing of the results.
        :param seed: (int, def=None) global seed for replicability
        :param logfile: (str, def=None) path to the logfile, if any, to store in the zip file.
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
        self.log.info('seed: ' + str(self.seed))

        # create output folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # auxiliar variables
        pass_to_next = None # Todo da rimuovere da questa classe
        ct_simulation = 0

        while ct_simulation < self.no_simulations:

            self.log.info("Running simulation {} / {}".format(ct_simulation + 1, self.no_simulations))

            try:
                # run simulation
                pass_to_next = self.run_single_simulation(pass_to_next=pass_to_next, **kwargs)
                # count simulation as correctly terminated
                ct_simulation += 1
                self.log.debug("status = completed")

            except self._controlled_exceptions as e:
                # here goas a list o
                # record failed simulation
                self.fails.append(ct_simulation)
                self.log.warning("status = failed: {}".format(e))

            assert len(self.fails) <= max_num_failure, 'We reached {} failed simulation.'.format(len(self.fails))

        self.serialise(self.folder + "/simulation.pkl")
        self.write_results()

        # Compress results in a .zip archive
        if logfile is not None:
            shutil.copy(logfile, self.folder + '/' + logfile)
        # command = 'zip -m ' + self.folder + '.zip ' + self.folder + ' -r '
        command = 'zip -m {0:s}.zip {0:s} -r'.format(self.folder)
        self.log.debug("executing: " + command)
        subprocess.Popen(command.split()).wait()

    def run_single_simulation(self, pass_to_next=None):
        raise cdg.utils.AbstractMethodError()

    def string_raw_results(self):
        return 'changes_true = ' + str(self.changes_true) + '\nchanges_est = ' + str(self.changes_est)

    def process_results(self, figures_merit=None):
        assert figures_merit in self._figure_merit_list.keys(), 'figure of merit {} not in the list.'.format(figures_merit)

    def write_results(self, figures_merit=None):
        """
        This function receives a list a figures of merit, compute them and
        print them out to a predefined file.

        :param figureMerit: list of figure of merit to be computed
        :return:
        """

        # open output file
        f = open(self.folder + self._exp_result_and_setup_file, 'w')

        # parameters and settings of the simulation
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Setting           # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines(self.folder + '\n')
        f.writelines(str(self.pars))
        # f.writelines(str(self.dataset))
        f.writelines('\n')
        f.writelines('number of concluded simulation:' + str(self.no_simulations) + '\n')
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
        fm, result, printout = self.process_results(figures_merit=figures_merit)
        f.writelines(printout + '\n')

        # close file
        f.close()

        self.log.info([fm, result])
        return [fm, result]

    @staticmethod
    def sequence_generator(dataset, pars):
        raise cdg.utils.AbstractMethodError()


class SimulationCDT(Simulation):
    """ Environment for assessing the performance of a CDT. """

    def __init__(self):
        # self.skip_from_serialising(['cpm'])
        super().__init__()
        raise NotImplementedError()
    
    @classmethod
    def figure_merit_list(cls):
        cls._figure_merit_list['dc'] = ['dc_rate', 'dc_rate_95ci']
        cls._figure_merit_list['dca'] = ['dca_rate', 'dca_rate_95ci']
        cls._figure_merit_list['arl'] = ['arl0', 'arl0_95ci', 'arl1', 'arl1_95ci']
        cls._figure_merit_list['fa1000'] = ['fa1000_rate', 'fa1000_rate_std']

        cls._figure_merit_list['dcaarl'] = cls._figure_merit_list['dca'] + cls._figure_merit_list['arl']
        cls._figure_merit_list['tnnls'] = cls._figure_merit_list['dca'] + cls._figure_merit_list['arl'] + cls._figure_merit_list['fa1000']

        cls._figure_merit_list['mat'] = ['matlab']
        return cls._figure_merit_list

    def empty_results(self):
        self.runLen0 = []
        self.runLen1 = []

    def run_single_simulation(self, pass_to_next=None, **kwargs):
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
        cusum = self.training_part(x_train, self.pars, threshold=pass_to_next['threshold'])

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

        return {'threshold': cusum.threshold}

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


    @classmethod
    def sequence_embedding(cls, embedding_space, dataset,
                           sequence_train, sequence_test, no_train_for_embedding=0,
                           message=None):
        raise cdg.util.errors.CDGAbstractMethod()

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

    def process_results(self, figures_merit='dca'):
        """
        Process the outcomes for synthetic results.
        :param figures_merit: list of figures of merit to be printed
        """

        super().process_results(figures_merit=figures_merit)

        printout = ''
        result = {}
        
        raise NotImplementedError('please fix according to SimCPM editions and fixes')

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
            # # Footer message
            printout += '\n\nAll went well, apparently.\n\n'

            return self._figure_merit_list[figures_merit], result, printout

        except IndexError:
            self.log.warning('something went wrong in computing the results.')
            printout += '....something went wrong....'
            return None, None, printout


class SimulationCPM(Simulation):
    """ Environment for assessing the performance of a CPM. """

    changes_true = None
    changes_est = None
    cpm = None
    use_fwer = None

    def __init__(self, cpm=None, use_fwer=True):
        """
        :param cpm: instance of class cdg.changedetection.CPM
        """
        # self.skip_from_serialising(['cpm'])
        super().__init__()
        self.cpm = cpm
        self.use_fwer = use_fwer

    @classmethod
    def figure_merit_list(cls):
        cls._figure_merit_list['discr'] = ['discr', 'discr_iqr']
        # cls._figure_merit_list['discrn'] = ['discrn', 'discrn_iqr']
        # # cls._figure_merit_list['ari'] = ['ari', 'ari_iqr']
        # cls._figure_merit_list['tpr'] = ['tpr', 'tpr_iqr']
        # cls._figure_merit_list['fpr'] = ['fpr', 'fpr_iqr']
        cls._figure_merit_list['tpr'] = ['tpr', 'tpr_95ci']
        cls._figure_merit_list['fpr'] = ['fpr', 'fpr_95ci']
        cls._figure_merit_list['ari'] = ['ari', 'ari_95ci']
        cls._figure_merit_list['rte'] = ['discrn', 'discrn_95ci']
        cls._figure_merit_list['cps'] = cls._figure_merit_list['tpr'] + cls._figure_merit_list['fpr']
        cls._figure_merit_list['tsp'] =  cls._figure_merit_list['cps'] + cls._figure_merit_list['ari'] + cls._figure_merit_list['rte']
        cls._figure_merit_list['tspiq'] = ['tpr', 'tpr_iqr'] + ['tpr', 'tpr_iqr'] + ['ari', 'ari_iqr'] + ['discrn', 'discrn_iqr']
        cls._figure_merit_list['tsp90'] = ['tpr', 'tpr_90ci'] + ['tpr', 'tpr_90ci'] + ['ari', 'ari_90ci'] + ['discrn', 'discrn_90ci']
        return cls._figure_merit_list

    def empty_results(self):
        self.changes_true = []
        self.changes_est = []

    def run_single_simulation(self, pass_to_next=None, **kwargs):

        self.log.debug('sequence generation...')
        return_indices = self.dataset.has_prec_distance()
        g_train, g_test, change_points = self.sequence_generator(dataset=self.dataset, pars=self.pars,
                                                                 return_indices=return_indices)
        self.changes_true.append(change_points)

        self.log.debug('embedding...')
        self.pars.embedding_method.fit(graphs=g_train, dist_fun=self.dataset.distance_measure, **kwargs)
        x = self.pars.embedding_method.transform(data=g_test)

        self.log.debug('operating phase...')
        self.cpm.reset()
        change_points_est, _ = self.cpm.predict(x=x, alpha=self.pars.significance_level,
                                                margin=self.pars.margin, fwer=self.use_fwer, 
                                                verbose=not self.use_fwer, **kwargs)
        self.changes_est.append(change_points_est)

        return change_points_est


    @classmethod
    def sequence_generator(cls, dataset, pars, return_indices=True):
        # parse possible None parameters
        lenghts = [len(dataset.elements[c]) for c in pars.classes] if pars.subseq_lengths_t is None else pars.subseq_lengths_t
        ratios = [1. for _ in pars.classes] if pars.subseq_ratios is None else pars.subseq_ratios

        # init
        sequence_train = []
        sequence_test = []
        changes = []

        # browse all subsequence
        for i in range(len(pars.classes)):
            # bootstrap lenghts[i] objects from the class[i] and lenghts[i] from the whole data set
            inclass = np.random.choice(dataset.elements[pars.classes[i]], size=lenghts[i])
            outclass = np.random.choice(dataset.get_all_elements(), size=lenghts[i])
            # decide which element to keep between inclass and outclass
            sub_division = np.random.rand(lenghts[i]) <= ratios[i]
            sequence = np.empty(inclass.shape, dtype=int)
            sequence[sub_division] = inclass[sub_division]
            sequence[np.logical_not(sub_division)] = outclass[np.logical_not(sub_division)]
            # split into train and test
            len_train = int(pars.train_len_ratio * lenghts[i])
            sequence_train += [s for s in sequence[:len_train]]
            sequence_test += [s for s in sequence[len_train:]]
            # update change-point list
            changes.append(len(sequence_test))
        # remove the last (which is not used)
        changes.pop(-1)

        if return_indices:
            return sequence_train, sequence_test, changes
        else:
            g_train = dataset.get_graphs(sequence_train, format=['cdg'])[0]
            g_test = dataset.get_graphs(sequence_test, format=['cdg'])[0]
            return g_train, g_test, changes

    def process_results(self, figures_merit='discr'):
        """
        Process the outcomes for synthetic results.
        :param figures_merit: list of figures of merit to be printed
        """

        super().process_results(figures_merit=figures_merit)

        printout = ''
        result = {}
        ff = '{0:.3f}' # float string format

        no_changes_sim = len(self.changes_true[0])
        no_changes_tot = no_changes_sim * self.no_simulations
        ch_true_mat = np.array(self.changes_true)
        # ch_est_mat = np.zeros(ch_true_mat.shape)

        # ------------------------------ #
        # Adjusted Rand index
        # ------------------------------ #
        ari = []
        for n in range(self.no_simulations):
            # ground-truth label of each data point
            label_true = np.zeros((self.pars.test_len_t))
            for i in range(no_changes_sim-1):
                label_true[self.changes_true[n][i]: self.changes_true[n][i+1]] = i + 1
            if no_changes_sim >= 1:
                label_true[:self.changes_true[n][-1]:] = no_changes_sim

            # predicted label of each data point
            label_pred = np.zeros((self.pars.test_len_t))
            for i in range(len(self.changes_est[n])-1):
                label_pred[self.changes_est[n][i]: self.changes_est[n][i+1]] = i + 1
            if len(self.changes_est[n]) >= 1:
                label_pred[:self.changes_est[n][-1]:] = len(self.changes_est[n])

            # Compute the ARI
            ari.append(sklearn.metrics.adjusted_rand_score(label_true, label_pred))

        # ------------------------------ #
        # Discrepancy
        # ------------------------------ #
        discr = []
        for i in range(self.no_simulations):
            if no_changes_sim < 1:
                for ch in self.changes_est[i]:
                    discr.append(min([ch, self.pars.test_len_t - ch]))
            else:
                for ch in self.changes_est[i]:
                    discr.append( min(np.fabs(ch - ch_true_mat[i, :])) )

        # discrepancy normalized on the length of the sequence
        discrn = [1.*d/self.pars.test_len_t for d in discr]

        if len(discr) == 0:
            # This issue occurs when no true change is present and 
            # no change is detected as well. 
            # todo Check for better solutions
            discr = [np.nan]
            discrn = [np.nan]


        # ------------------------------ #
        # False positive rate and true positive rate
        # ------------------------------ #

        fp_list = np.full(self.no_simulations, np.nan)   # number of discovered false positive
        tp_list = np.full(self.no_simulations, np.nan)   # number of discovered true positive
        for i in range(self.no_simulations):
            fp_list[i] = max([0., len(self.changes_est[i]) - no_changes_sim])
            tp_list[i] = min([len(self.changes_est[i]), no_changes_sim])


        # ------------------------------ #
        # Compute estimates and confidence intervals
        # ------------------------------ #

        prc_list = [2.5, 5, 25, 50, 75, 95, 97.5]

        # Continuous statistics
        tmp = {}
        for el in ['discr', 'discrn', 'ari']:
            exec("tmp[0] = {}".format(el))
            result[el] = ff.format(np.mean(tmp[0]))
            result[el + '_std'] = ff.format(np.std(tmp[0]))
            for pp in prc_list:
                result['{}_{:03d}'.format(el, round(pp*10))] = ff.format(scipy.percentile(tmp[0], pp))

        # Discrete statistics
        if no_changes_sim == 0:
            assert np.all(tp_list == 0.)
        elif no_changes_sim == 1:
            assert np.all(tp_list >= 0.) and np.all(tp_list <= 1.)
        result['tpr'] = ff.format(np.mean(tp_list))
        result['fpr'] = ff.format(np.mean(fp_list))
        result['tpr_std'] = ff.format(np.std(tp_list))
        result['fpr_std'] = ff.format(np.std(fp_list))
        for pp in prc_list:
            if no_changes_sim == 0:
                tmp = 0.0
            elif no_changes_sim == 1:
                tmp = bernoulli_mean_percentile(tp_list, pp)
            else:
                tmp = discrete_mean_percentile(tp_list, pp)
            result['tpr_{:03d}'.format(round(pp*10))] = ff.format(tmp)
            tmp = discrete_mean_percentile(fp_list, pp)
            result['fpr_{:03d}'.format(round(pp*10))] = ff.format(tmp)

        # Some pre-formatted results
        tmp = {}
        for el in ['discr', 'discrn', 'ari', 'fpr', 'tpr']:
            result[el + '_iqr'] = '[{}, {}]'.format(result[el + '_250'], result[el + '_750'])
            result[el + '_90ci'] = '[{}, {}]'.format(result[el + '_050'], result[el + '_950'])
            result[el + '_95ci'] = '[{}, {}]'.format(result[el + '_025'], result[el + '_975'])

            printout += el + ' :  mean (std) [i.q. range] = '
            printout += '{} ({}) {}\n'.format(result[el], result[el + '_std'], result[el + '_iqr'])

            printout += el + ' :  mean (std) [90 conf.int.] = '
            printout += '{} ({}) {}\n'.format(result[el], result[el + '_std'], result[el + '_90ci'])

            printout += el + ' :  mean (std) [95 conf.int.] = '
            printout += '{} ({}) {}\n'.format(result[el], result[el + '_std'], result[el + '_95ci'])


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

        # # Footer message
        printout += '\n\nAll went well, apparently.\n\n'
        return self._figure_merit_list[figures_merit], result, printout

def bernoulli_mean_percentile(x, prc):
    quantile_order = prc * 0.01
    if isinstance(x, list):
        N = len(x)
    elif isinstance(x, np.ndarray):
        N = x.shape[0]
    else:
        raise NotImplementedError("Array of type {} not handled".format(type(x)))
    assert np.all(x>=0.0) and np.all(x<=1.0)

    p = np.mean(x)
    if p == 0.0 or p == 1.0:
        # ISSUE: for p=0 scipy.binom.ppf return n-1
        # https://github.com/scipy/scipy/issues/1603
        # https://github.com/scipy/scipy/issues/5122
        # This piece of code is a work around
        return p
    return scipy.stats.binom.ppf(q=quantile_order, n=N, p=p, loc=0) / N

def discrete_mean_percentile(x, prc):
    quantile_order = prc * 0.01
    if isinstance(x, list):
        N = len(x)
    elif isinstance(x, np.ndarray):
        N = x.shape[0]
    else:
        raise NotImplementedError("Array of type {} not handled".format(type(x)))
    x_mu = np.mean(x)
    x_std = np.std(x)
    if x_std == 0:
        return x_mu
    return scipy.stats.norm.ppf(q=quantile_order, loc=x_mu, scale=x_std/np.sqrt(N))

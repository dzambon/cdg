# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Repeated simulations of a graph stream and a change detection test on it.
# See also [1] and [2].
#
#
# References:
# ---------
# [1] Zambon, Daniele, Cesare Alippi, and Lorenzo Livi. "Concept Drift and 
#     Anomaly Detection in Graph Streams." arXiv preprint arXiv:1706.06941 (2017). 
#     Submitted.
#
# [2] Zambon, Daniele, Lorenzo Livi, and Cesare Alippi. "Detecting Changes in 
#     Sequences of Attributed Graphs." IEEE Symposium Series on Computational 
#     Intelligence. 2017.
#
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 17/09/2017
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import numpy as np
from scipy import percentile
from scipy.stats import binom
from cdg import graph
from cdg import changedetection
import sys
import random
import datetime
import os
import subprocess
import pickle


class Simulation():

    def __init__(self):
        self.p = None
        self.dataset = None
        self.numSimulations = None

    def set(self, parameters, dataset, numSimulations):
        self.pars = parameters
        self.dataset = dataset
        self.numSimulations = numSimulations

        # load dataset
        print('retrieving precomputed dissimilarities...')
        self.dataset.loadGraphNameMap()
        self.dataset.loadDissimilarityMatrix()


    def run(self, main_seed=None):
        """
        Run repeated simulations and perform some processing of the results.
        :param main_seed: (None) global seed for replicability
        :return: run lengths under H0, run lengths under H1
        """

        # create output folder
        if not os.path.exists(self.pars.folder):
            os.makedirs(self.pars.folder)

        # set global seed
        if main_seed is None:
            main_seed = random.randint(0,10000)
        np.random.seed(main_seed)
        random.seed(main_seed)

        self.main_seed = main_seed
        self.seeds =[]

        # init result variables
        detected_changes = 0
        self.runLen0 = []
        self.runLen1 = []
        self.fails=[]

        # auxiliar variables
        h=None
        n=0


        while n<self.numSimulations:

            # local seed for the current simulation
            seed = random.randint(0,10000)
            np.random.seed(seed)
            random.seed(seed)
            self.seeds.append(seed)

            print("\n***  Running simulation %d / %d" % (n+1, self.numSimulations))
            print("seed = %d" % (seed))

            # filename and title of the output figure
            self.pars.title_tmp = "(s:%d) <%s> vs <%s>" % (seed, self.pars.nameNominal, self.pars.nameNonnominal)
            self.pars.filename_tmp = "%s_%s_n%d_s%d" % ( self.pars.nameNominal, self.pars.nameNonnominal, n, seed)


            try:

                # run simulation
                if h is None:
                    h, runLen0_tmp, runLen1_tmp = self.single_simulation(threshold=None)
                else:
                    _, runLen0_tmp, runLen1_tmp = self.single_simulation(threshold=h)

                # store estimated times
                self.runLen0.append(runLen0_tmp)
                self.runLen1.append(runLen1_tmp)

                # count simulation as correctly terminated
                n += 1

                print("\nstatus = completed")

            except np.linalg.LinAlgError:
                # record failed simulation
                self.fails.append((n,seed))
                print("\nstatus = failed (singular matrix?)")



        self.save_raw_results() 
        self.processRawResults() 

        # # Compress results in a .zip archive

        command = 'zip '+self.pars.folder+'.zip '+self.pars.folder+' -r'
        print("executing: " + command)
        subprocess.Popen(command.split()).wait()


    def single_simulation(self, threshold=None):
        """
        Run a single simulation
        :param threshold: (None) a predefined threshold for the cusum, otherwise estimated
        :return: adopted_theshold, list runLen0, delay of detection
        """
        # setup the dataset; prototype selection; embedding
        y_train, y = self.dataset_embedding_part()
        cusum = self.training_part(y_train, threshold=threshold)

        # Launch operating phase
        runLen0, runLen1 = self.operating_phase(cusum,y)
        return cusum.estimated_h, runLen0, runLen1



    def stream_generation(self):
        """
        Bootstrap the graphs, select the prototypes and compute all the dissimilarities.
        :return: y_train for training the cusum, y for the operating phase
        """


        streamTraining = self.dataset.generateBootstrappedStream(self.pars.class0, self.pars.trainTotal_t())
        streamTest_0 = self.dataset.generateBootstrappedStream(self.pars.class0, self.pars.testNominal_t)
        streamTest_1 = self.dataset.generateBootstrappedStream(self.pars.class1, self.pars.testNonnominal_t)

        if self.pars.testDrift_t>0:

            prcStreamClass = []

            for t in range(0,self.pars.testDrift_t):
       
                prc_t = 1.*t/self.pars.testDrift_t
        
                prcClass = []
                for c in self.pars.class0:
                    prcClass.append(prc_t)    
                for c in self.pars.class1:
                    prcClass.append(1 - prc_t)

                prcStreamClass.append(prcClass)
        
            streamTest_d = self.dataset.generateBootstrappedStream(self.pars.allClasses(), self.pars.testDrift_t, prc=prcStreamClass)
            
        else:
            streamTest_d = []


        streamTest = streamTest_0 +streamTest_d+streamTest_1

        return streamTraining, streamTest







    def operating_phase(self,cusum, y):
        """
        Launch the cusum on the windowed data
        :param cusum: cusum instance
        :param y: all test data, numPrototyeps \times numTestWindow
        :return:
        """

        orrible_trick_for_plots = False


        # # Auxiliar variables

        # variables for run lengths
        runLen0 = []
        runLen1 = []

        # vars for change time
        ctWin = 0  # window unit
        ctTime = 0  # time unit
        alarmWin = -1
        alarmTime = -1
        changeTime = int(np.floor(self.pars.testNominal_t))-1
        changeWin = changeTime//self.pars.winSize
        changeOccurred = False

        statusBarLen = 70
        noTotalWin =  int(np.floor(self.pars.testNominal_t) + np.floor(self.pars.testNonnominal_t)) //self.pars.winSize
        statusBarStep = int(np.floor(1.* noTotalWin / statusBarLen))

        # vars for graphics
        gg = []  # for plotting
        incc = []  # for plotting


        # # # Change detection -- CUSUM

        # # Init cusum

        cusum.reset()
        alarm=False
        # loop
        while True:


            # check whether a change occurred
            if ctWin == changeWin:
                changeOccurred = True
                sys.stdout.write('|')
            # status bar
            if ctWin%statusBarStep == 0:
                if alarmWin>0:
                    sys.stdout.write('*')
                elif alarm:
                    sys.stdout.write("'")
                else:
                    sys.stdout.write('.')


            # # Iterate

            alarm, inc = cusum.iterate(y[:, ctTime: ctTime + self.pars.winSize], reset=False)

            # output stuff
            for i in range(self.pars.winSize):
                incc.append(inc)
                gg.append(cusum.g)


            # # Checks about changes

            if alarm:
                if not changeOccurred:
                    runLen0.append(cusum.time)
                else:
                    if len(runLen1)==0:
                        alarmWin = ctWin
                        alarmTime = ctTime
                        runLen1.append(ctWin-changeWin)
                    else:
                        runLen1.append(cusum.time)

                if not orrible_trick_for_plots  :
                    cusum.reset()

            # # Time update

            ctWin += 1
            ctTime += self.pars.winSize
            if ctTime >= self.pars.testNominal_t + self.pars.testNonnominal_t:
                break


        return runLen0, runLen1



    def save_raw_results(self):

        # Save as pickle file.
        pickleDictionary = {}

        for key in self.__dict__:
            if key != 'dataset':
                pickleDictionary[key]=self.__dict__[key]
        pickle.dump(pickleDictionary, open(self.pars.folder+"/simulation.pkl", "wb" ) )


    def processRawResults(self,figureMerit=['dca_rate']):

        # self.loadPickleDictionary()

        # # # Process the results

        # open output file
        f = open(self.pars.folder+"/000_experiment_setup_and_results.txt", 'w')

        # # Description of the experiment

        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Setting           # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')

        f.writelines(self.pars.folder + '\n')

        f.writelines(str(self.pars))
        f.writelines(str(self.dataset))
        f.writelines('\n')

        f.writelines('number of concluded simulation:'+str(self.numSimulations) + '\n')
        f.writelines('main seed:'+str(self.main_seed) + '\n')


        # # Raw results of the simulations -- run lengths

        f.writelines('\n\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Raw results       # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')

        f.writelines('runLen0 = '+str(self.runLen0) + '\n')
        f.writelines('runLen1 = '+str(self.runLen1) + '\n')
        f.writelines('failed simulations:'+str(self.fails) + '\n')

        # # Processed results of the simulations
        exit, tabularResult = self.process_results(f=f, figureMerit=figureMerit)
        if(exit):
            # # Footer message
            f.writelines('\n\n')
            f.writelines('# Footer #\n')
            f.writelines('all went well, apparently')
            f.writelines('\n\n')

        f.close()

        return tabularResult



    def process_results(self, f, figureMerit):
        """
        Process the outcomes for synthetic results.
        :param f: file handler for the output
        """

        result={}


        # process alarms
        try:
            meanRunLen0, falseAlarms, cleanMeanRunLen0 = process_alarms(self.runLen0, None)
            meanRunLen1, trueAlarms,  cleanMeanRunLen1 = process_alarms(self.runLen1, None)
            # meanRunLen1, trueAlarms, _  = processAlarms(runLen1, -1) 
        except ValueError as e:
            print("ooops penso che runLen0 o runLen1 sia vuoto... Error: ")
            print(e)

        if len(meanRunLen0) != len(meanRunLen1) or len(meanRunLen0) != self.numSimulations:
            raise ValueError('len(meanRunLen0) != len(meanRunLen1)...')




        # # Processed results of the simulations

        f.writelines('\n\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')
        f.writelines('# # # Processed results # # #' + '\n')
        f.writelines('# # # # # # # # # # # # # # #' + '\n')


        try:



            # Detected Change Rate (DCR) (equivalently : Test hp: observed ARL1 = target ARL0)
            try:                
                dc = 0 
                for s in range(0,self.numSimulations):
                    if trueAlarms[s] == 0:
                        dc += 0 # do nothing
                    elif meanRunLen1[s] < self.pars.arl_w:
                        dc += 1
                dc_rate = 1.*dc/self.numSimulations
                dc_rate_std = binom_se(dc_rate,self.numSimulations)
                dc_rate_a, dc_rate_b = binom_ci95(dc_rate,self.numSimulations)
            except ValueError as e:
                print("penso che trueAlarms sia vuoto... Error: ")
                print(e)
                dc_rate = dc_rate_std = dc_rate_a = dc_rate_b = -1
            f.writelines('detected changes rate :  DCR (std) [95 conf.int.] = %.3f (%.3f) [%.3f, %.3f]\n' % (dc_rate, dc_rate_std, dc_rate_a, dc_rate_b) )
            result['dc_rate']       = '%.3f' % dc_rate
            result['dc_rate_std']   = '%.3f' % dc_rate_std
            result['dc_rate_a']     = '%.3f' % dc_rate_a
            result['dc_rate_b']     = '%.3f' % dc_rate_b
            result['dc_rate_95ci']  = '[%s, %s]' % (result['dc_rate_a'], result['dc_rate_b'])

            # Test hp: observed ARL0 = target ARL0
            try:
                rl_bi = []
                for s in range(0,self.numSimulations):
                    if falseAlarms[s]==0:
                        rl_bi.append(0)
                    elif meanRunLen0[s] > self.pars.arl_w:
                        rl_bi.append(0)
                    else:
                        rl_bi.append(1)

                rl0_bi_p = sum(rl_bi) * 1. / self.numSimulations
                rl0_bi_a, rl0_bi_b = binom_ci95(rl0_bi_p, self.numSimulations)
            except ValueError as e:
                print("penso che runLen0 sia vuoto... Error: ")
                print(e)
                rl0_bi_p = rl0_bi_a = rl0_bi_b = -1
            f.writelines('test per ARL0 : mean [95 c.i.] = %.3f [%.0f, %.0f]\n' % (rl0_bi_p, rl0_bi_a, rl0_bi_b))
            result['rl0_bi_p'] = '%.3f' % rl0_bi_p
            result['rl0_bi_a'] = '%.3f' % rl0_bi_a
            result['rl0_bi_b'] = '%.3f' % rl0_bi_b
            result['arl0_bi_95ci']  = '[%s, %s]' % (result['rl0_bi_a'], result['rl0_bi_b'])


            # Detected Change Rate adapted (DCRa) (equivalently : Test hp: observed ARL1 = observed ARL0)
            try:
                dca = 0 
                for s in range(0,self.numSimulations):
                    if trueAlarms[s] == 0:
                        dca += 0 # do nothing
                    elif falseAlarms[s] == 0:
                        dca += 1
                    elif meanRunLen1[s] < meanRunLen0[s]:
                        dca += 1
                dca_rate = 1.*dca/self.numSimulations
                dca_rate_std = binom_se(dca_rate,self.numSimulations)
                dca_rate_a, dca_rate_b = binom_ci95(dca_rate,self.numSimulations)
            except ValueError as e:
                print("penso che meanRunLen1 sia vuoto... Error: ")
                print(e)
                dca_rate = dca_rate_std = dca_rate_a = dca_rate_b = -1
            f.writelines('detected changes rate adapted:  DCRa (std) [95 conf.int.] = %.3f (%.3f) [%.3f, %.3f]\n' % (dca_rate, dca_rate_std, dca_rate_a, dca_rate_b) )
            result['dca_rate']      = '%.3f' % dca_rate
            result['dca_rate_std']  = '%.3f' % dca_rate_std
            result['dca_rate_a']    = '%.3f' % dca_rate_a
            result['dca_rate_b']    = '%.3f' % dca_rate_b
            result['dca_rate_95ci'] = '[%s, %s]' % (result['dca_rate_a'], result['dca_rate_b'])


            # Average Run Length 0
            try:
                arl0 = np.mean(cleanMeanRunLen0)
                rl0_00 = min(cleanMeanRunLen0)
                rl0_025 = percentile(cleanMeanRunLen0,2.5)
                rl0_05 = percentile(cleanMeanRunLen0,5)
                rl0_25 = percentile(cleanMeanRunLen0,25)
                rl0_50 = percentile(cleanMeanRunLen0,50)
                rl0_75 = percentile(cleanMeanRunLen0,75)
                rl0_95 = percentile(cleanMeanRunLen0,95)
                rl0_975 = percentile(cleanMeanRunLen0,97.5)
                rl0_100 = max(cleanMeanRunLen0)
            except ValueError as e:
                print("penso che runLen1 sia vuoto... Error: ")
                print(e)
                arl0 = rl0_00 = rl0_025 = rl0_05 = rl0_25 = rl0_50 = rl0_75 = rl0_95 = rl0_975 = rl0_100 = -1
            f.writelines('average run length in the nominal regime : ARL0 [IQ interval] = %.0f [%.0f, %.0f]\n' % (arl0, rl0_25, rl0_75) )
            f.writelines('average run length in the nominal regime : [0.00,.05,.25,.50,.75,.95,1.00] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n' % (rl0_00, rl0_05, rl0_25, rl0_50, rl0_75, rl0_95, rl0_100) )
            result['arl0']      = '%.0f' % arl0
            result['rl0_00']    = '%.0f' % rl0_00
            result['rl0_025']   = '%.0f' % rl0_025
            result['rl0_05']    = '%.0f' % rl0_05
            result['rl0_25']    = '%.0f' % rl0_25
            result['rl0_50']    = '%.0f' % rl0_50
            result['rl0_75']    = '%.0f' % rl0_75
            result['rl0_95']    = '%.0f' % rl0_95
            result['rl0_975']   = '%.0f' % rl0_975
            result['rl0_100']   = '%.0f' % rl0_100
            result['arl0_95ci'] = '[%s, %s]' % (result['rl0_025'], result['rl0_975'])


            # Average Run Length 1  (Delay of Detection)
            try:
                arl1 = np.mean(cleanMeanRunLen1)
                rl1_00 = min(cleanMeanRunLen1)
                rl1_025 = percentile(cleanMeanRunLen1,2.5)
                rl1_05 = percentile(cleanMeanRunLen1,5)
                rl1_25 = percentile(cleanMeanRunLen1,25)
                rl1_50 = percentile(cleanMeanRunLen1,50)
                rl1_75 = percentile(cleanMeanRunLen1,75)
                rl1_95 = percentile(cleanMeanRunLen1,95)
                rl1_975 = percentile(cleanMeanRunLen1,97.5)
                rl1_100 = max(cleanMeanRunLen1)
            except ValueError as e:
                print("penso che runLen1 sia vuoto... Error: ")
                print(e)
                arl1 = rl1_00 = rl1_025 = rl1_05 = rl1_25 = rl1_50 = rl1_75 = rl1_95 = rl1_975 = rl1_100 = -1
            f.writelines('average run length in the out-of-control regime : DoD [IQ interval] = %.0f [%.0f, %.0f]\n' % (arl1, rl1_25, rl1_75) )
            f.writelines('average run length in the out-of-control regime : [0.00,.05,.25,.50,.75,.95,1.00] = [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n' % (rl1_00, rl1_05, rl1_25, rl1_50, rl1_75, rl1_95, rl1_100) )
            result['arl1']      = '%.0f' % arl1
            result['rl1_00']    = '%.0f' % rl1_00
            result['rl1_025']   = '%.0f' % rl1_025
            result['rl1_05']    = '%.0f' % rl1_05
            result['rl1_25']    = '%.0f' % rl1_25
            result['rl1_50']    = '%.0f' % rl1_50
            result['rl1_75']    = '%.0f' % rl1_75
            result['rl1_95']    = '%.0f' % rl1_95
            result['rl1_975']   = '%.0f' % rl1_975
            result['rl1_100']   = '%.0f' % rl1_100
            result['arl1_95ci'] = '[%s, %s]' % (result['rl1_025'], result['rl1_975'])

            
            # False alarms within 1000 samples
            try:
                scaleFactor = 1000 / (self.pars.testNominal_t) 
                fa1000_rate = np.mean(falseAlarms) * scaleFactor
                fa1000_rate_std = np.std(falseAlarms) * scaleFactor
                fa1000_rate_25 = percentile(falseAlarms,25) *scaleFactor
                fa1000_rate_75 = percentile(falseAlarms,75) *scaleFactor
            except ValueError as e:
                print("penso che runLen1 sia vuoto... Error: ")
                print(e)
                fa1000_rate = fa1000_rate_std = fa1000_rate_25 = fa1000_rate_75 = -1
            f.writelines('false alarms rate per 1000 sample : FA1000 (std) [IQ interval] = %.3f (%.3f) [%.0f, %.0f]\n' % (fa1000_rate, fa1000_rate_std, fa1000_rate_25, fa1000_rate_75) )
            result['fa1000_rate'] = fa1000_rate
            result['fa1000_rate_std'] = fa1000_rate_std
            result['fa1000_rate_25'] = fa1000_rate_25
            result['fa1000_rate_75'] = fa1000_rate_75


            # # Latex table entries

            f.writelines('\n\n')
            f.writelines('# # # # # # # # # # # # # # #' + '\n')
            f.writelines('# # # Latex table entry # # #' + '\n')
            f.writelines('# # # # # # # # # # # # # # #' + '\n')


            closingString = ' \\\\'
            try:
                closingString += '\t % %s\n' % self.dataset.notes
            except:
                closingString += '\n'





            if figureMerit[0] == 'matlab':
                figureMerit = ['{[dca rate ,  dca rate a ,  dca rate b , no Prot ,  win Size] ,  "M..n.._name"}']
                selectedResults = ["{[%f,%f,%f], 'M=%d, n=%s'}, ... %s" % (dca_rate,dca_rate_a,dca_rate_b,self.pars.noPrototypes,self.pars.winSize,self.dataset)]
            else:
                selectedResults = []
                for figMer in figureMerit:
                    selectedResults.append(result[figMer])

            return True, [figureMerit, selectedResults]

        except IndexError:
            f.writelines('....something went wrong....')
            return False, None


    def loadPickleDictionary(self, pickleFile=None):

        if pickleFile is None:
            pickleFile = self.pars.folder+"/simulation.pkl"

        pickleDict = pickle.load(open(pickleFile , "rb" ))
        self.savePickleDictionary(pickleDict)

        return pickleDict

    def savePickleDictionary(self, pickleDict):
        for key in pickleDict:
            self.__dict__[key] = pickleDict[key]
















class SimulationTLC(Simulation):


    def training_part(self, y_train, threshold=None):
        # Train the cusum
        cusum = changedetection.TLCCusum(arl=self.pars.arl_w)
        if threshold is None:
            cusum.training(y_train, beta=self.pars.beta, estimate_threshold=True, \
                numSimulations=self.pars.noSimulationsThreshEstimation, \
                lenSimulations=self.pars.lenSimulationThreshEstimation)
        else:
            cusum.training(y_train, beta=self.pars.beta, estimate_threshold=False)
            cusum.estimated_h=threshold

        return cusum
        

    def dataset_embedding_part(self):
        """
        Bootstrap the graphs, select the prototypes and compute all the dissimilarities.
        :return: y_train for training the cusum, y for the operating phase
        """

        streamTrain, streamTest = self.stream_generation()

        # # Prototype Selection

        diss_matrix_prot_sel = self.dataset.subDissimilarityMatrix(streamTrain[:self.pars.trainProtSel_t] , streamTrain[:self.pars.trainProtSel_t])
        # select the prototypes
        numAnnealing=20
        val=np.inf
        for ct in range(0,numAnnealing):
            prot_id_tmp, val_tmp = graph.Prototype.k_centers(diss_matrix_prot_sel, self.pars.noPrototypes)
            if ct==0 or val_tmp<val:
                prot_id=list(prot_id_tmp)
                val = val_tmp
        prototypes = [streamTrain[i] for i in prot_id]


        # # Embed the training graphs

        y_train = self.dataset.subDissimilarityMatrix(prototypes,streamTrain[self.pars.trainProtSel_t:])
     
        # # Embed all test observations in dissimilarity space

        print('embedding...')
        y = self.dataset.subDissimilarityMatrix(prototypes,streamTest)


        return y_train, y




















class SimulationStdErr(Simulation):

    def training_part(self, y_train, threshold=None):

        # Train the cusum
        self.pars.winSize=1
        cusum = changedetection.StdErrCusum(arl=self.pars.arl_w)
        cusum.training(y_train, beta=self.pars.beta, estimate_threshold=True)

        return cusum



    def dataset_embedding_part(self):
        """
        Bootstrap the graphs, select the prototypes and compute all the dissimilarities.
        :return: y_train for training the cusum, y for the operating phase
        """

        streamTrain, streamTest = self.stream_generation()



        # # Compute feature of (Embed) the training graphs

        method = self.pars.noPrototypes
        sys.stdout.writelines("\n")

        ct = 0
        y_train = []
        y = []
        for g in (list(streamTrain) + list(streamTest)):

            el = self.dataset.getNameAndClass(g)
            if el is None:
                raise ValueError("something went wrong with the retrieval of the graph")
            graphName = el[1]
            
            if method==0:
                val=graph.Graph(graphName,self.dataset.path).get_density()
            elif method==-1:
                val=graph.Graph(graphName,self.dataset.path).get_laplacian_spec_gap() 

            sys.stdout.writelines("       \rComputing graph feature: %d" % ct)    
            if ct < len(streamTrain):  
                y_train.append(val)
            else:
                y.append(val)

            ct += 1
     
        print()

        if ( len(y_train) != len(streamTrain) )  or ( len(y) != len(streamTest) )  :
            raise ValueError("something went wrong in storing the feature")


        return np.array([y_train]), np.array([y])












def binom_se(p, n):
    return np.sqrt(p * (1 - p) / n)


def binom_ci95(p, sim):
    # ISSUE: for p=0 scipy.binom.ppf return n-1
    # https://github.com/scipy/scipy/issues/1603  
    # https://github.com/scipy/scipy/issues/5122
    # This code is work around to by pass it
    if p==0:
        a = 0.0
        b = 0.0
    elif p==1:
        a = 1.0
        b = 1.0
    else:
        a = binom.ppf(q=.025, n=sim, p=p, loc=0) / sim
        b = binom.ppf(q=.975, n=sim, p=p, loc=0) / sim

    return a, b

def process_alarms(runLen, default=-1):
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




class Parameters:

    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True



        
    def __init__(self):


        self.timeCreation = datetime.datetime.now() 

        # Problem

        self.class0 = None
        self.class1 = None


        # Method

        self.noPrototypes = 3
        self.winSize = 15
        
        self.beta = .75

        self.arl_w = 100


        # Stream

        self.trainProtSel_t = 300 
        self.trainCusum_t   = 1000 

        self.testNominal_t    = self.arl_t()*4
        self.testDrift_t      = self.arl_t()*0
        self.testNonnominal_t = self.arl_t()*2

        self.noSimulationsThreshEstimation = 10000
        self.lenSimulationThreshEstimation = self.testTotal_w() + 1


        # Output

        self.folder = self.timeCreation.strftime('demo_%G%m%d_%H%M_%S')
        self.nameNominal    = 'n.d.'
        self.nameNonnominal = 'n.d.'

        self.launchingCommand = None

        self.title_tmp = None
        self.filename_tmp = None

        self._freeze()




    def arl_t(self):
        return self.arl_w*self.winSize

    def trainTotal_t(self):
        return self.trainProtSel_t + self.trainCusum_t 
    def trainTotal_w(self):
        return self.trainTotal_t() //self.winSize

    def testTotal_t(self):
        return self.testNominal_t + self.testDrift_t + self.testNonnominal_t
    def testTotal_w(self):
        return self.testTotal_t()//self.winSize

    def setDriftChange(self,prc=0.5):
        self.testDrift_t      = int(self.testNominal_t * prc)
        self.testNominal_t   -= self.testDrift_t

    def allClasses(self):
        return self.class0 + self.class1        

    def __str__(self):
        string = ''
        for key in self.__dict__.keys():
            string += "%s = %s\n" % (key,self.__dict__[key])

        return string

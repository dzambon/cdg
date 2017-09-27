# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# CUSUM-based change detection tests. See also [1] and [2].
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
from scipy.stats import chi2
import sys



class Cusum:
    """
    """

    def __init__(self, arl=np.inf):
        """
        """
        # CUSUM parameters
        self.gamma = 0
        self.time = 0
        self.g = 0
        self.arl = arl
        self.h = np.inf
        self.estimated_h = None

    def reset(self,time=0, g=0):
        self.time = time
        self.g = g


    def iterate(self,  measured_statistic=-1, reset=False):
        # # Update from the Previous Step
        
        # update h
        if self.estimated_h is None:
            self.h = np.inf
        elif self.time < len(self.estimated_h):
            self.h = self.estimated_h[self.time]
        else:
            self.h = self.estimated_h[-1]
        
        # check if reached the threshold
        if self.g > self.h:
            alarm = True
            if reset:
                self.reset()
        else:
            alarm = False

        
        # # Next Step
        
        # increment time
        self.time += 1
    
        # update g
        increment = measured_statistic - self.gamma

        self.g += increment
        if self.g < 0:
            self.g = 0

        # return the increment
        return alarm, increment

    def training(self, training_data, beta=.75, gamma_type='quantile',estimate_threshold=False):

        if training_data.shape[1]<2:
            # print(training_data)
            raise ValueError('empty training data')

        # gamma
        self.estimate_gamma(training_data, beta=beta, gamma_type=gamma_type)

        # estimate threshold
        if estimate_threshold:
            self.estimate_threshold(training_data)
            
    def estimate_gamma(self,training_data, beta=.75, gamma_type='quantile'):
        
        if gamma_type=="quantile":
            self.gamma = percentile(training_data,int(beta*100))
        elif gamma_type == "std":
            self.gamma = np.mean(training_data) + beta*std(training_data)
        else:
            raise ValueError("tipo di gamma non riconosciuto")


    def estimate_threshold(self,training_data):
        alpha = 100. - 100/self.arl
        g = []
        for stat in training_data[0]:
            # print("s_" +str(stat))
            self.iterate(measured_statistic=stat)
            g.append(self.g)

        self.reset()
        self.estimated_h = np.array([percentile(g,alpha)])




class TLCCusum(Cusum):

    # def __init__(self, use_d2=False, arl=1000):
    def __init__(self, arl=1000):

        Cusum.__init__(self,arl=arl)

        # self.use_d2=use_d2

        # data parameters
        self.training_sample_size = 0
        self.sample_dim = 0

        # H0 null hp parameters
        self.mu_0 = 0
        self.s2_0 = 0
        self.s2_0inv = 0


    def iterate(self, y, reset=False, simulated_d2=-1):
        """
        :param y: test sample
        :param y: test sample
        :return: d2 - gamma 
        """
        # compute increment
        if simulated_d2 < 0:
            # d2 = self.hotelling_t_squared_test(y)
            d2 = self.chi2test(y)
            # if not self.use_d2:
            #     d2 = np.sqrt(d2)
        else:
            d2 = simulated_d2
        
        alarm, increment = Cusum.iterate(self,reset=reset,measured_statistic=d2)
        # return the increment
        return alarm, increment


    def chi2test(self, y):

        # # Assess Mahalanobis' Distance

        # mu = mu_1 - mu_0
        mu = np.mean(y, axis=1) - self.mu_0

        # Mahalanobis' distance
        d2 = np.dot(mu, np.dot(self.s2_0inv, mu))
        d2 /= 1/self.training_sample_size + 1/y.shape[1]

        return d2

    def training(self, y_training, beta=.75, gamma_type='quantile', estimate_threshold=False, numSimulations=10000, lenSimulations=1000):
        self.sample_dim = y_training.shape[0]
        self.training_sample_size = y_training.shape[1]

        # initialize mu_0
        self.mu_0 = np.mean(y_training, axis=1)
        
        if self.training_sample_size>1:
            # initialize S2_0
            self.s2_0 = np.cov(y_training)  # this is the unbiased version

            # initialize S2_0inv
            if self.sample_dim == 1:
                self.s2_0inv = 1 / self.s2_0
            else:
                self.s2_0inv = np.linalg.inv(self.s2_0)

        # gamma
        self.estimate_gamma(beta=beta, gamma_type=gamma_type)
        # print("gamma= " + str(self.gamma))

        # estimate threshold
        if estimate_threshold:
            self.estimate_threshold( numSimulations=numSimulations)
            # self.estimate_threshold( numSimulations=numSimulations, lenSimulations=lenSimulations)


    def estimate_gamma(self, beta=.75, gamma_type='quantile'):
        
        if gamma_type=="quantile":
            self.gamma = chi2.ppf(beta,df=self.sample_dim)
        elif gamma_type == "std":
            self.gamma = chi2.mean(df=self.sample_dim) + beta*chi2.std(df=self.sample_dim)
        else:
            raise ValueError("tipo di gamma non riconosciuto")
        # if not self.use_d2:
        #     self.gamma = np.sqrt(self.gamma)


    def estimate_threshold(self, numSimulations):


        print("estimating threshold...")

        maxLenSimulations = 2000
        S=numSimulations

        # percentile of data to be retained
        percentile = (1- 1./self.arl)*100

        # init
        g=np.zeros(S)
        h=[]

        # time cycle
        for n in range(0,maxLenSimulations):

            if 1.*S/numSimulations< .5:
                print("too many dropped simulations... ")
                break


            #number of simulation active
            S=len(g)

            # next step
            d2 = chi2.rvs(self.sample_dim,size=S)
            # if not self.use_d2:
            #     d2 = np.sqrt(d2)

            g = g+d2-self.gamma

            # estimate h
            h.append(np.percentile(g,percentile))

            # check simulated data
            rem = []
            for s in range(0,S):
                if g[s] < 0:
                    # 0 bound
                    g[s]=0
                elif g[s] > h[n]:
                    # component exceeding the threshold
                    rem.append(s)

            # delete alarmed simulation
            g=np.delete(g,rem)

            # status bar
            sys.stdout.write("\r" + str(n))
            sys.stdout.flush()

        sys.stdout.write("\n")
        print("survived simulations: %d / %d" %(S , numSimulations) )

        meanStart = 50
        if len(h)<meanStart:
            meanSize=-1 # ??
        h[-1]=np.mean(h[meanStart:-1])
        self.estimated_h = np.zeros((1)) + h[-1]


class StdErrCusum(Cusum):
    """
    The class StdErrCusum extends Cusum. It consider the bilateral monitoring of 
    the deviation of a generic statistic from its mean value. Specifically, it monitors
    the standard error = abs( y - mean).
    """

    def __init__(self, arl=1000):
        self.mean=0
        Cusum.__init__(self,arl=arl)


    def iterate(self, measured_statistic, reset=False ):
        """
        :param y: test sample
        :return: d2 - gamma 
        """

        if type(measured_statistic) is np.ndarray:
            measured_statistic = measured_statistic[0,0]
        # compute increment
        stdErr = np.abs(self.mean - measured_statistic)
        
        alarm, increment = Cusum.iterate(self,measured_statistic=stdErr, reset=reset)
        # return the increment
        return alarm, increment
    
    def training(self, training_data, beta=.75, gamma_type='quantile',estimate_threshold=False):

        self.mean = np.mean(training_data)

        # gamma
        training_se = np.zeros(training_data.shape) + training_data - self.mean
        for i in range(0,training_data.shape[0]): 
            for j in range(0,training_data.shape[1]): 
                training_se[i,j] = np.abs(training_data[i,j] - self.mean)
        Cusum.estimate_gamma(self, training_se, beta=beta, gamma_type=gamma_type)

        # estimate threshold
        if estimate_threshold:
            Cusum.estimate_threshold(self,training_data)


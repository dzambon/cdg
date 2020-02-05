# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Implements several two-sample tests.
# --------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import scipy.stats

import cdg.utils


def energy_distance(D, t):
    '''
    Energy distance:
      Székely, Gábor J., and Maria L. Rizzo.  
      Energy statistics: A class of statistics based on distances.  
      Journal of statistical planning and inference, 2013.  

    :param D: square dissimilarity matrix
    :param t: split point so that 0:t is the first sample and t: is the second one.
    :return:
        - squared energy distance rescaled by t*(T-t)/T
        - squared energy distance
    '''
    T = D.shape[0]
    if t < 1 or t >= T:
        raise ValueError("argument t must be in {1, T-1}, as the indexing start from 0.")
    D10 = np.sum(D[:t][:, t:]) / t / (T - t)
    D00 = np.sum(D[:t][:, :t]) / t / t
    D11 = np.sum(D[t:][:, t:]) / (T - t) / (T - t)
    energy2 = 2 * D10 - D00 - D11
    return energy2 * t * (T - t) / T, energy2

def mmd2u(K, t):
    '''
    :param K: square kernel matrix
    :param t: split point so that 0:t is the first sample and t: is the second one.
    :return:
        - squared mmd
    '''
    T = K.shape[0]
    if t < 1 or t >= T:
        raise ValueError("argument t must be in {1, T-1}, as the indexing start from 0.")
    K10 = K[:t][:, t:]
    K00 = K[:t][:, :t]
    K11 = K[t:][:, t:]

    s10 = K10.sum() / ((t) * (T-t))
    s00 = (K00.sum() - K00.diagonal().sum()) / ((t) * (t-1))
    s11 = (K11.sum() - K11.diagonal().sum()) / ((T-t) * (T-t-1))
    mmd2 = s00 + s11 - 2 * s10
    return mmd2


class PairwiseMeasureTest(object):
    '''
    Tests based on a pairwise (dis)similarity measure.
    The current class handles the possibility of computing the measure only when needed
    by storing the distance function.
    '''
    measure_function = None
    
    def __init__(self, measure_fun=None, take_sets=True):
        self.measure_function = measure_fun
        self.take_sets = take_sets
        
    def measure(self, x1, x2):
        if self.take_sets:
            return self.measure_function(x1, x2)
        else:
            n1 = len(x1) if isinstance(x1, list) else x1.shape[0]
            n2 = len(x2) if isinstance(x2, list) else x2.shape[0]
            mea_mat = -np.ones((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    mea_mat[i, j] = self.measure_function(x1[i], x2[j])
            return mea_mat


class PWDistanceBasedTest(PairwiseMeasureTest):
    def __init__(self, distance_fun=None):
        super().__init__()
        self.measure_function = distance_fun

    @property
    def distance_measure(self):
        return self.measure_function


class PWKernelBasedTest(PairwiseMeasureTest):
    def __init__(self, kernel_fun=None):
        super().__init__()
        self.measure_function = kernel_fun
    
    @property
    def kernel_measure(self):
        return self.measure_function


class TwoSampleTest(cdg.utils.Loggable, cdg.utils.Pickable):
    """
    Wrapper for two-sample statistical hypothesis tests.
    """
    name = 'GenericTwoSampleTest'

    def __init__(self):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)

    @classmethod
    def predict(cls, x0, x1, alpha, **kwargs):
        """
        Run the test passing two separeted samples.
        :param x0: (n0, d) n0 realizations of the first d-dimensional sample.
        :param x1: (n1, d) n1 realizations of the second d-dimensional sample.
        :param alpha: significance level of the inference.
        :param kwargs:
        :return:
            - pval: p-value of the test, or True if pval < alpha
            - stat: value of the statistic
            - th: threshold associated with the statistic
        """
        raise cdg.utils.AbstractMethodError()
    
    @classmethod
    def predicts(cls, x, t, alpha, **kwargs):
        """
        Same test as `fit` but passing a single sequence of data and a split index.
        :param x: (n, d) sample of size n and dimensional d.
        :param t: splitting index so that x0 = x[:t] and x1 = x[t:] are the two samples.
        :param alpha: significance level of the inference.
        :param kwargs:
        :return:
            - pval: p-value of the test, or True if pval < alpha
            - stat: value of the statistic
            - th: threshold associated with the statistic
        """
        raise cdg.utils.AbstractMethodError()


class Student_t_test(TwoSampleTest):
    name = 't-test'
    
    @classmethod
    def predict(cls, x0, x1, alpha=0.05, **kwargs):
        correction = kwargs.pop('correction', 0)
        d = x0.shape[1]
        n0 = x0.shape[0]
        n1 = x1.shape[0]
        cov0 = np.cov(x0.transpose())
        cov1 = np.cov(x1.transpose())
        cov_pulled = (n0 * cov0 + n1 * cov1) / (n0 + n1 - 2)
        if d == 1:
            cov_inv = 1. / cov_pulled
        else:
            cov_inv = np.linalg.inv(cov_pulled)
        delta_mu = np.mean(x0, axis=0) - np.mean(x1, axis=0)
        # stat = (n0 + n1) * np.dot(np.dot(delta_mu, cov_inv), delta_mu)
        # pval = 1 - scipy.stats.chi2.cdf(stat+correction, df=d)
        # n0 n1 / n0+n1  mahal \sim T2 (d, n0+n1 -2)
        # n+1-p / pn T2(p,n) = F(p,n-p)
        # n0+n1-1-d / (d(n0+n1-2)) T2(d,n0+n1-2) = F(d,n0+n1-2-d)
        # n0 n1 / n0+n1  mahal \sim T2 (d, n0+n1 -2)
        # n0 n1 / n0+n1 * n0+n1-1-d / (d(n0+n1-2)) mahal = F(d,n0+n1-2-d)
        mahal = np.dot(np.dot(delta_mu, cov_inv), delta_mu)
        stat = (n0 * n1) / (n0 + n1) * (n0 + n1 - 1 - d) / (d * (n0 + n1 - 2)) * mahal
        pval = 1 - scipy.stats.f.cdf(stat + correction, dfn=d, dfd=n0 + n1 - 2 - d)
        th = scipy.stats.f.ppf(q=1 - alpha, dfn=d, dfd=n0 + n1 - 2 - d)
        return pval, stat, th
    
    @classmethod
    def predicts(cls, x, t, alpha=0.05, **kwargs):
        x0 = x[:t]
        x1 = x[t:]
        return cls.predict(x0=x0, x1=x1, alpha=alpha, **kwargs)


class MeanDistanceScore(TwoSampleTest, PWDistanceBasedTest):
    name = 'MuDist'
    
    def __init__(self, distance_fun=None):
        PWDistanceBasedTest.__init__(self, distance_fun=distance_fun)
        TwoSampleTest.__init__(self)

    @classmethod
    def predict(cls, x0, x1, alpha=0.05, **kwargs):
        x = np.concatenate((x0, x1), axis=0)
        t = x0.shape[0]
        return cls.predicts(x=x, t=t, alpha=alpha, **kwargs)
    
    def predicts(self, x, t, alpha=0.05, **kwargs):
        """
        See super class...
        :param kwargs:
            - is_dist_mat: whether or not x contains already the square distance matrix
            - distance_fun: function that compute the distance between any x[i, :] and x[j, :]
        """
        is_dist_mat = kwargs.pop('is_dist_mat', False)
        # distance_fun = self.get_distance_fun(**kwargs)
        dist_mat = x if is_dist_mat else self.distance(x, x, symmetric=True)
    
        if self.distance_measure is None:
            delta_mu = np.mean(x[:t], axis=0) - np.mean(x[t:], axis=0)
            stat = np.dot(delta_mu, delta_mu)
        else:
            losses1 = np.sum(dist_mat[:t] ** 2, axis=0)
            mu1i = np.argmin(losses1)
            losses2 = np.sum(dist_mat[t:] ** 2, axis=0)
            mu2i = np.argmin(losses2)
            stat = dist_mat[mu1i, mu2i] ** 2
        
        return 1, stat, 0


class EnergyTest(TwoSampleTest, PWDistanceBasedTest):
    '''
    Energy test for equality of distribution based on
      Székely, Gábor J., and Maria L. Rizzo.  
      Energy statistics: A class of statistics based on distances.  
      Journal of statistical planning and inference, 2013.  
    '''
    
    name = 'Energy'
    
    def __init__(self, distance_fun=None, repetitions=None):
        PWDistanceBasedTest.__init__(self, distance_fun=distance_fun)
        TwoSampleTest.__init__(self)
        self.repetitions = repetitions

    def predict(self, x0, x1, alpha=0.05, **kwargs):
        x = np.concatenate((x0, x1), axis=0)
        t = x0.shape[0]
        return self.predicts(x=x, t=t, alpha=alpha, **kwargs)
    
    def predicts(self, x, t, alpha=0.05, **kwargs):
        repetitions = kwargs.pop('repetitions', self.repetitions)
        correction = kwargs.pop('correction', 0)
        is_dist_mat = kwargs.pop('is_dist_mat', False)
        n_jobs = kwargs.pop('n_jobs', 1)

        dist_mat = x if is_dist_mat else self.distance(x, x, symmetric=True)

        return self._test_dist_mat(dist_mat, t, alpha=alpha,
                                   repetitions=repetitions, correction=correction,
                                   n_jobs=n_jobs, **kwargs)
    
    @classmethod
    def _test_dist_mat(cls, dist_mat, t, alpha, repetitions, correction, n_jobs, **kwargs):
        
        def test_fun(x):
            stat, _ = energy_distance(D=x, t=t)
            return stat
        
        stat = test_fun(dist_mat)
        verbose = kwargs.pop('verbose', True)
        if repetitions == 0:
            pval = 1
        else:
            pval = cdg.utils.permutation_test(x=dist_mat, test_fun=test_fun,
                                              observed=stat, repetitions=repetitions,
                                              is_matrix=True, verbose=verbose,
                                              n_jobs=n_jobs)
        threshold = None
        return pval, stat, threshold


class MMDTest(TwoSampleTest, PWKernelBasedTest):
    '''
    MMD test for equality of distribution.
    '''
    
    name = 'MMD'
    
    def __init__(self, kernel_fun=None, repetitions=None):
        PWKernelBasedTest.__init__(self, kernel_fun=kernel_fun)
        TwoSampleTest.__init__(self)
        self.repetitions = repetitions
    
    def predict(self, x0, x1, alpha=0.05, **kwargs):
        x = np.concatenate((x0, x1), axis=0)
        t = x0.shape[0]
        return self.predicts(x=x, t=t, alpha=alpha, **kwargs)
    
    def predicts(self, x, t, alpha=0.05, **kwargs):
        repetitions = kwargs.pop('repetitions', self.repetitions)
        correction = kwargs.pop('correction', 0)
        is_kernel_mat = kwargs.pop('is_kernel_mat', False)
        n_jobs = kwargs.pop('n_jobs', 1)

        kernel_mat = x if is_kernel_mat else self.kernel(x, x, symmetric=True)
        
        return self._test_kernel_mat(kernel_mat, t, alpha=alpha,
                                     repetitions=repetitions, correction=correction, n_jobs=n_jobs)

    @classmethod
    def _test_kernel_mat(cls, kernel_mat, t, alpha, repetitions, correction, n_jobs):
        def test_fun(x):
            stat = mmd2u(K=x, t=t)
            return stat
    
        stat = test_fun(kernel_mat)
        pval = cdg.utils.permutation_test(x=kernel_mat, test_fun=test_fun, observed=stat,
                                          repetitions=repetitions, is_matrix=True, n_jobs=n_jobs)
        threshold = None
        return pval, stat, threshold


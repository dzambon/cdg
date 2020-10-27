# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Defines the basis structure of the change-detection tests and the 
# change-point methods
# --------------------------------------------------------------------------------
import numpy as np
import scipy
from tqdm import tqdm
import joblib

import cdg.utils
import cdg.changedetection

def cusum_alarm_curve(cusum, sequence, y=None, arl=None, **kwargs):
    """

    :param cusum: the used instance of Cusum, if None a new is created.
    :param sequence: univariate sequence of statistics
    :param y_true: ground-truth labels
    :param arl: (None) if not None, the procedure stops once the a specific arl is reached
    :param kwargs:
        - verbose: verbosely print info to stdout
    :return:
        - alarm_curve: which is a list of triples
                [..., (false_alarms, true_alarms, threshold). ...]
            with `false_alarms` and `true_alarms` obtained with `threshold`
            If arl is not None, only the corresponding triple is returned.
    """
    # seq = np.array(sequence) if isinstance(sequence, list) else sequence
    # assert seq.ndim==1 or seq.shape[1]==0
    verbose = kwargs.pop('verbose', False)
    seq = sequence
    y_true = np.zeros((sequence.shape[0], 1)) if y is None else y
    
    if np.any(y_true):
            raise NotImplementedError('specific arl = {} can be computed only with '
                                      'a sequence entirely in nominal regime.')
    max_false_alarms = np.inf if arl is None else 1. * sequence.shape[0] / arl

    if hasattr(cusum, 'window_size'):
        y_true = y_true[::cusum.window_size]
        max_false_alarms /= cusum.window_size

    alarm_curve = []
    threshold = None  # Start with infinite threshold bc/ Cusum checks for >=
    max_g = np.array([np.inf])
    false_alarms = 0
    true_alarms = 0
    pbar = tqdm(leave=True, total=y_true.shape[0], disable=not verbose, desc='ARL curve est. (up.bound niter)')
    while false_alarms < max_false_alarms \
            and max_g > 0 \
            and max_g != threshold:
        threshold = max_g
        cusum.reset()
        cusum.threshold = threshold
        y_pred, g = cusum.predict(sequence, reset=True, window_result=True)
        # true_alarms = np.sum(np.logical_and(y_pred, y_true))
        false_alarms = np.sum(y_pred) - true_alarms
        alarm_curve.append((false_alarms, true_alarms, threshold))
        # max_g = max(g)
        assert (g<threshold).sum() > 0, "probably g never reached 0."
        max_g = g[g<threshold].max()
        # print(len(np.where(g<threshold)[0]))
        pbar.update()
    pbar.close()

    if arl is None:
        return alarm_curve
    else:
        return alarm_curve[-1:]

def cusum_arl0_curve(cusum, sequence, arl=None, **kwargs):
    """

    :param cusum: Cusum instance to run
    :param sequence: sequence of statistics
    :param y_true: ground-truth labels
    :param arl: (None) if not None, the procedure stops once the a specific arl is reached
    :param kwargs:
        - verbose: verbosely print info to stdout
    :return:
        - arl0_curve: which is a list of triples
                [..., (average_run_length, threshold). ...] with `average_run_length`
            obtained with `threshold`
            If arl is not None, only the corresponding triple is returned.

    """
    no_datapoints = sequence.shape[0]
    y_true = np.full((no_datapoints,1), False)
    alarm_curve = cusum_alarm_curve(cusum=cusum, sequence=sequence,
                                    y_true=y_true, arl=arl, **kwargs)
    arl0_curve = []
    arl_previous = -1
    for ac in alarm_curve:
        if ac[0] > 0:
            arl_current = int(1.*no_datapoints/ac[0])
            if arl_current != arl_previous:
                arl0_curve.append((arl_current, ac[-1]))
            arl_previous = arl_current

    return arl0_curve

def _save_arl0_curve(arl0_curve, beta, dof):

    inc = [1] * 10 + [10] * 19 + [100] * 9
    runlengths = [sum(inc[:i]) for i in range(1, len(inc))]

    small_list = []
    for ac in arl0_curve:
        if ac[0] in runlengths:
            small_list.append(ac)
    print( '...[({},{})] = {}'.format(beta, dof, small_list))
    
class ChangeDetectionTest(cdg.utils.Loggable, cdg.utils.Pickable):
    def __init__(self):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)

class Cusum(ChangeDetectionTest):
    """
    Super class implementing the basis of the change detection tests introduced in
        Concept Drift and Anomaly Detection in Graph Streams
        Daniele Zambon, Cesare Alippi and Lorenzo Livi
        IEEE Transactions on Neural Networks and Learning Systems, 2018.
    Despite the name `Cusum`, this is not the classical version, rather
    it's only based on the cumulative sum chart.
    """

    def __init__(self, arl=np.inf, beta=.75):
        """
        :param arl: target Average Run Length to be reproduced by the threshold.
            Default is np.inf.
        :param beta: sensitivity parameter in interval (0,1).
        """
        super().__init__()
        self.gamma = 0
        self.time = 0
        self.g = 0
        self.arl = arl
        self.threshold = np.inf
        self.sample_dim = None
        self.beta = beta
        self.log.debug("Cusum created")

    def fit(self, x, estimate_threshold=False, **kwargs):
        """
        :param x: (no_train, d) training data.
        :param estimate_threshold: whether or not to estimate the threshold.
            Default is False.
        :return: True if the procedure completed as expected, False otherwise
        """
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            raise TypeError('x has to be (no_train, d) numpy array')
        self.sample_dim = x.shape[1]
        # gamma
        gamma_type = kwargs.pop('gamma_type', 'quantile')
        res1 = self._estimate_gamma(x, gamma_type=gamma_type, **kwargs)
        # estimate threshold
        res2 = True
        if estimate_threshold is True:
            threshold_type = kwargs.pop('threshold_type', 'numerical')
            res2 = self._estimate_threshold(x=x, threshold_type=threshold_type, **kwargs)
        # clean
        self.reset()
        return res1 and res2

    def _estimate_gamma(self, x, gamma_type, **kwargs):
        res = True
        if gamma_type == "quantile":
            gamma = scipy.percentile(x, int(self.beta * 100))
        elif gamma_type == "std":
            gamma = np.mean(x) + self.beta * np.std(x)
        else:
            res = False
            raise ValueError(
                'gamma_type <{}> not available.'.format(str(gamma_type)))
        self.gamma = gamma
        return res

    def _estimate_threshold(self, x, **kwargs):
        """

        :param x:
        :param kwargs:
            - precompute_thresholds: this is only works for precomputing the
                thresholds to be stored (hardcoded) for speed up future runs.
            - dof: if precompute_thresholds is true, this is required. It is the
                number of degrees of freedom.
        :return:
        """
        assert x.ndim == 1 or x.shape[1] == 1
        
        # check whether to use precomputed thresholds
        precompute_thresholds = kwargs.pop('precompute_thresholds', False)
        if precompute_thresholds:
            dof = kwargs.pop('dof', None)
            if dof is None:
                raise ValueError('you didnt provide the number of dof')
            self.log.info('computing arl0 curve; this may take a while.')
            arl0_curve = cusum_arl0_curve(self, sequence=x, arl=self.arl, **kwargs)
            _save_arl0_curve(arl0_curve, beta=self.beta, dof=dof)
            return True

        self.log.info('computing arl0 curve; this may take a while.')
        arl0_curve = cusum_arl0_curve(self, sequence=x, arl=self.arl, **kwargs)

        threshold = arl0_curve[0][-1]
        if threshold == 0:
            self.threshold = np.inf
            g, _ = self.predict(x)
            th_current = max(g)
            self.log.warn("th_current = {}. Is the training set too small?".format(str(th_current)))
            res = False
        elif threshold == np.inf or threshold is None:
            self._estimate_threshold(x, **kwargs)
            raise cdg.utils.ImpossibleError("current threshold = {}".format(str(threshold)))
        else:
            res = True

        self.threshold = threshold

        return res

    def reset(self, time=0, g=0):
        self.time = time
        self.g = g
        return self

    def iterate(self, datum):
        """
        Iterates the change detection test of one step.
        The current implementation deals only with scalar datapoints.

        :param datum: the scalar datum (a statistic) measured at the current
            time. If a np.array is passed it considers only the first component.
        :param reset: flag, whether to reset the accumulator every time an alarm
            is raised.
        :return alarm: `True` if an alarm is raised, `False` otherwise,
        :return increment: the new increment for cumulative sum.
        """
        if isinstance(datum, np.ndarray):
            if datum.ndim == 1:
                measured_statistic = datum[0]
            else:
                measured_statistic = datum[0, 0]
        else:
            measured_statistic = datum

        # increment time
        self.time += 1

        # update g
        increment = measured_statistic - self.gamma
        self.g += increment
        if self.g < 0:
            self.g = 0

        # check if reached the threshold
        alarm = True if self.g >= self.threshold else False

        # return the increment
        return alarm, increment

    def predict(self, x, reset=True, continued=False, **kwargs):
        """
        Runs method `Cusum.iterate` for the entire testing phase.

        :param x: (no_test, ) test data (ideally, the stream of data).
        :param reset: flag, whether to reset the accumulator every time an alarm
            is raised.
        :param continued: (bool, def=False) whether to continue from the previous state, or reset.
        :param verbose: if True the progress bar is printed
        :return y_predict: (no_test, ) array of predictions: True if alarm.
        :return cumulative_sums: (no_test, ) array g(t).
        """
        verbose = kwargs.pop('verbose', False)
        
        g = []
        num_alarms = 0
        if not continued:
            self.reset()
        alarms = []
        for i in tqdm(range(x.shape[0]), disable=not verbose):
            alarm_tmp, _ = self.iterate(datum=x[i])
            alarms.append(alarm_tmp)
            g.append(self.g)
            if alarm_tmp:
                num_alarms += 1
                if reset:
                    self.reset()

        y_predict = np.array(alarms)[:, None]
        cumulative_sums = np.array(g)[:, None]

        return y_predict, cumulative_sums


class WindowedAndMultivariateCusum(Cusum):
    """
    Ideally an abstract class extending `Cusum` to deal with data stream
    that has to be windowed.  It deals also with multivariate data.

    The user passes a sequence
        x[0],  x[1],  x[2],  ..., x[n],  ...
    and the current mechanism applies a change detection test on a second
    sequence
        x'[0], x'[1], x'[2], ..., x'[w], ...
    where
        - x'[w] = stat( x[ w*win_size : (w+1)*winw_size] )
        - `win_size` is the width of the window
        - `stat(.)` is any statistic of the window, e.g., the mean.

    Regarding multivariate streams, method `compute_local_statistic` is
    supposed to return a scalar quantity, in fact, `Cusum` works on a scalar
    stream.
    """

    def __init__(self, arl=100, window_size=1, beta=.75):
        self.window_size = window_size
        super().__init__(arl=arl, beta=beta)

    def compute_data_windows(self, x):
        n = (x.shape[0] // self.window_size) * self.window_size
        x_w = x[:n].reshape(-1, self.window_size, x.shape[1])
        return x_w
    
    def predict(self, x, reset=True, window_result=False, **kwargs):
        """
        The same algorithm of the super-method, but splitting the testing data
        `x` into windows of size `self.window_size`.
        :return y_predict: (no_test, ) = (no_windows * window_size, ) array of predictions: True if alarm.
        :return cumulative_sums: (no_test, ) = (no_windows * window_size, ) array g(t).
        """
        # compute statistics in each window
        x_win = self.compute_data_windows(x=x)
        # run univariate cusum
        y_pred_w, cum_sum_w = super().predict(x_win, reset=reset, **kwargs)
        # reshape results
        if window_result:
            return y_pred_w, cum_sum_w
        else:
            one_vec = np.ones((1, self.window_size))
            y_pred_t = y_pred_w.dot(one_vec).reshape(-1, 1)
            cum_sum_t = cum_sum_w.dot(one_vec).reshape(-1, 1)
            return y_pred_t, cum_sum_t

    def compute_local_statistic(self, x_win):
        raise cdg.utils.AbstractMethodError()

    def iterate(self, datum):
        # compute statistics in each window
        assert datum.shape[0] == self.window_size
        s_win = self.compute_local_statistic(x_win=datum[None, ...])
        return super().iterate(datum=s_win)

    def _estimate_gamma(self, x, gamma_type):
        # compute the local statistics sw
        self.gamma=0
        sw = self.compute_local_statistic(x)
        # _, gw = self.predict(x, reset=False, window_result=True)
        # sw = gw[1:] - gw[:-1]
        # estimate gamma
        return super()._estimate_gamma(sw, gamma_type='quantile')

    def _estimate_threshold(self, x, threshold_type='data', **kwargs):
        # sw = self.compute_local_statistic(x)
        _, _, th = cusum_alarm_curve(cusum=self, sequence=x, arl=self.arl, verbose=True)[0]
        self.threshold = np.array([th])
        return True

class MultiChangePointMethod(cdg.utils.Loggable, cdg.utils.Pickable):
    """
    General framework for the identification of multiple change points.
    """
    name = 'GenericMultiCPM'
    
    _cpm_attr_to_reset = ['cps',               'pvals',
                          'cps_fwer',          'pvals_fwer',
                          'cp_last_candidate', 'pval_last_candidate',
                          'stats_seq', 'pvals_seq', 'ths_seq']
    
    @property
    def cps_sorted(self):
        return sorted(self.cps)
    
    def __init__(self):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        self.reset()
    
    def predict(self, x, alpha, margin, **kwargs):
        raise cdg.utils.errors.CDGAbstractMethod()
    
    def reset(self):
        for a in self._cpm_attr_to_reset:
            self.__dict__[a] = None

class ChangePointMethod(MultiChangePointMethod):
    """
    General framework for Change-Point Method (CPM) as an extension of the MultiCPM.
    """
    
    name = 'GenericCPM'
    
    def __init__(self, local_test=None, name=None, **kwargs):
        """
        :param local_test: instance of cdg.changedetection.cpm.TwoSampleTest.
        :param name:
        :param kwargs:
            - use_max_stat: (True)
            - verbose: (True) True/False to enable tqdm progress bars
            - correction: (None) use to estimate quantiles in permutation tests.
            - repetitions: number of repetitions of the permutation tests.
            - n_jobs: (1) number of jobs for joblib; if n_jobs=1, joblib is not used
        """
        super().__init__()
        if local_test is not None:
            assert isinstance(local_test, cdg.changedetection.TwoSampleTest)
            self.local_test = local_test
            self.name = 'CPM+{}'.format(local_test.name) if name is None else name
        else:
            self.name = 'CPM+{None}'

        self._use_max_stat = kwargs.pop('use_max_stat', True)
        self._tqdm_disable = not kwargs.get('verbose', True)
        self._correction = kwargs.pop('correction', None)
        self._n_jobs = kwargs.pop('n_jobs', 1)

        # self.keep_in_serialising(['local_test'])
    
    def predict(self, x, alpha, margin, fwer=True, **kwargs):
        """
        Run the CPM to estimate the change point
        :param x: (T, d) ordered sample of size T and dimension d
        :param alpha: significance level of the inference.
        :param margin: minimum number of indexes in each sample.
        :param fwer: (bool, def=True) whether to employ the family-wise error rate, or 
            the p-value of the local two-sample tests.
        :param kwargs:
            - use_max_stat: (True)
            - verbose: (True) True/False to enable tqdm progress bars
            - correction: (None) use to estimate quantiles in permutation tests.
            - repetitions: number of repetitions of the permutation tests.
            - n_jobs: (1) number of jobs for joblib; if n_jobs=1, joblib is not used
        :return:
            - t_hat: index of the change point
            - pvals: list of p-values associate with each test, or True if pval < alpha
            - stats: list of values of the statistic at each time index
            - ths: list of threshold associated with the statistics
        """
        use_max_stat = kwargs.pop('use_max_stat', self._use_max_stat)
        tqdm_disable = not kwargs.get('verbose', self._tqdm_disable)
        correction = kwargs.pop('correction', self._correction)
        # n_jobs = kwargs.pop('n_jobs', self._n_jobs)
        
        if use_max_stat:
            R_current = 0  # TODO this R_current should not be visible in this class
        else:
            R_current = kwargs.get('repetitions', None)
            if R_current is None or R_current < 1. / alpha:
                R_current = int(1.1 / alpha)
        T = x.shape[0]
        if correction is None:
            correction = np.zeros((T))
        # Even thought T+1-2*margin elements are sufficient, the indexing would become more complicated
        pvals = np.full((T), np.nan)
        stats = np.full((T), np.nan)
        ths = np.full((T), np.nan)

        kwargs_no_rep = kwargs.copy()
        kwargs_no_rep.pop('repetitions', False)
        
        for t in tqdm(range(margin, T + 1 - margin), disable=tqdm_disable, desc='cpm'):
            pval_t, stat_t, th_t = self.local_test.predicts(x=x, t=t, alpha=alpha,  correction=correction[t],
                                                            repetitions=R_current, **kwargs_no_rep)
            pvals[t] = pval_t
            stats[t] = stat_t
            ths[t] = th_t

        # identify the last one and compute the pvalue, in case
        if use_max_stat:
            t_hat = np.nanargmax(stats)
            kwargs['verbose']=True
            pval_hat, stat_hat, th_hat = self.local_test.predicts(x=x, t=t_hat, alpha=alpha,
                                                                  correction=correction[t_hat],
                                                                  **kwargs)
            pvals[t_hat] = pval_hat
            stats[t_hat] = stat_hat
            ths[t] = th_hat
        else:
            # this deals with multiple minima, instead of taking `t_hat = np.nanargmin(pvals)`
            t_list = np.where(pvals==np.nanmin(pvals))[0]
            t_mean = np.mean(t_list)
            t_hat = t_list[np.argmin(np.abs(t_list - t_mean))]
        self.pvals_seq = pvals
        self.stats_seq = stats
        self.ths_seq = ths
        
        if pvals[t_hat] <= alpha:
            self.cps = [t_hat]
            self.pvals = [pvals[t_hat]]
        else:
            self.cps = []
            self.pvals = []
        
        self.cp_last_candidate = t_hat
        self.pval_last_candidate = pvals[t_hat]
        
        if fwer:
            return self.get_fwer(T=T, alpha=alpha, margin=margin, **kwargs)
        else:
            return self.cps, self.pvals

    @classmethod
    def _bonferroni_alpha(cls, alpha, T, margin):
        return 1. * alpha / (T + 1 - 2 * margin)

    @classmethod
    def _bonferroni_fwer(cls, pval, T, margin):
        return 1. * pval * (T + 1 - 2 * margin)

    def get_fwer(self, T, alpha, margin, **kwargs):
        self.cps_fwer = []
        self.pvals_fwer = []
        self.pvals_fwer_seq = self._bonferroni_fwer(self.pvals_seq, T, margin)
        for i in range(len(self.pvals)):
            # fwer = self.pvals[i] * (x.shape[0] + 1 - 2 * margin)
            fwer = self._bonferroni_fwer(self.pvals[i], T, margin)
            if fwer <= alpha:
                self.cps_fwer.append(self.cps[i])
                self.pvals_fwer.append(fwer)
        return self.cps_fwer, self.pvals_fwer

    
class MultiCPM(MultiChangePointMethod):
    pass

class CPM(ChangePointMethod):
    pass


def test_cusum_alarm_curve():
    from tqdm import tqdm
    import numpy as np
    # np.random.seed(20190225)

    d=3
    n=10000
    arl=30

    mu = np.random.randn(d)
    sigma = np.eye(d) + np.random.rand(d, d)
    # Sigma += Sigma.transpose()

    from cdg.changedetection import GaussianCusum
    cdt = GaussianCusum(arl=arl, window_size=10)
    for i in range(2):

        x_train = mu + np.dot(np.random.randn(n, d), sigma.transpose())
        y_train = x_train[:, 0] * 0

        x = mu + np.dot(np.random.randn(n, d), sigma.transpose())
        y_true = x[:, 0] * 0
    
        if i == 0:
            cdt.fit(x, estimate_threshold=True)
            th_true = cdt.threshold
            gamma_true = cdt.gamma
        else:
            # cdt.fit(x, estimate_threshold=True, threshold_type='data')
            cdt.fit(x, estimate_threshold=True, gamma_type='data', threshold_type='data')
            print('thresholds:\ttrue={}\test={}'.format(th_true, cdt.threshold))
            # cdt.fit(x, gamma_type='data')
            # print(f'gamma:\ttrue={gamma_true}\test={cdt.gamma}')
        # th_true = cdt.threshold
        #
        # _, _, th = cusum_alarm_curve(cusum=cdt, sequence=x_train, arl=arl, y_true=y_train, verbose=True)[0]
    # cdt._mu_0 = mu
    # cdt._s2_0inv = np.linalg.inv(np.dot(sigma, sigma.transpose()))
    y_predict, cumulative_sums = cdt.predict(x, reset=True, verbose=False)

    print(np.sum(y_predict)/n)
   
if __name__ == '__main__':
    test_cusum_alarm_curve()
    # cusum_alarm_curve(sequence=np.random.permutation(1000), arl=100)

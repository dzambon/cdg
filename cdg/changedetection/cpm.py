# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implement several change-point methods.
# --------------------------------------------------------------------------------
import numpy as np
import scipy
import scipy.stats
import scipy.spatial.distance

from cdg.utils import EDivisiveRFileNotFoundError 
from cdg.changedetection import MultiChangePointMethod, ChangePointMethod, EnergyTest, MMDTest


class PermutationBasedCPM(ChangePointMethod):
    """
    CPM based on pairwise graph measures, like a distanc or a kernel.
    """
    
    name = 'pwCPM'
    
    def __init__(self, repetitions=None, **kwargs):
        """
        :param repetitions: repetitions of the permutation test.
        """
        super().__init__(**kwargs)
        self.repetitions = repetitions
    
    def predict(self, x, alpha, margin=1, fwer=True, **kwargs):
        repetitions = kwargs.pop('repetitions', self.repetitions)
        if fwer:
            b_alpha = self._bonferroni_alpha(alpha, x.shape[0], margin)
            if repetitions is None or 1 / (1 + repetitions) > b_alpha:
                repetitions = int(1.1 / b_alpha)
        else:
            if repetitions > 0 and repetitions < 1. / alpha:
                repetitions = int(1.1 / alpha)
        return super().predict(x=x, alpha=alpha, repetitions=repetitions, fwer=fwer, margin=margin, **kwargs)

class MuCPM(ChangePointMethod):
    def __init__(self):
        from cdg.changedetection import Student_t_test
        super().__init__(local_test=Student_t_test())

class EnergyCPM(PermutationBasedCPM):
    """
    CPM based on energy test.
    """

    name = 'ECPM'

    def __init__(self, repetitions=None, **kwargs):
        super().__init__(local_test=EnergyTest(repetitions=repetitions), name=type(self).name, **kwargs)
        self.repetitions = repetitions

    def predict(self, x, alpha, margin, **kwargs):
        is_dist_mat = kwargs.pop('is_dist_mat', False)
        if not is_dist_mat:
            dm = scipy.spatial.distance.cdist(x, x, metric='euclidean')
        else:
            dm = x
        return super().predict(x=dm, alpha=alpha, margin=margin, is_dist_mat=True, **kwargs)


class MMDCPM(PermutationBasedCPM):
    """
    CPM based on energy test.
    """

    name = 'mmdCPM'

    def __init__(self, repetitions=99):
        """
        :param repetitions: repetitions of the permutation test.
        """
        super().__init__(local_test=MMDTest(repetitions=repetitions), name=type(self).name)

    def predict(self, x, alpha=0.05, margin=1, **kwargs):
        is_kernel_mat = kwargs.pop('is_kernel_mat', False)
        if not is_kernel_mat:
            km = x.dot(x.T)
        else:
            km = x
        return super().predict(x=km, alpha=alpha, margin=margin, is_kernel_mat=True, **kwargs)


class EDivisive_R(MultiChangePointMethod):
    '''
    Python wrapper of the e-divisive test in R
      James, Nicholas A., and David S. Matteson.  
      ecp: An R package for nonparametric multiple change point analysis of multivariate data.  
      arXiv preprint arXiv:1309.3295. 2013.  
    '''

    name = 'EDivR'

    cp_last_candidate = None
    pval_last_candidate = None

    r_script = 'ediv_run.r'
    csv_data = 'data.csv'
    txt_result = 'results.txt'
    json_results = 'results.json'

    def __init__(self, repetitions=None):
        """
        :param repetitions: repetitions of the permutation test.
        """
        super().__init__()
        self.repetitions = repetitions

    def clear_files(self):
        import os
        if os.path.isfile(self.r_script):
            os.remove(self.r_script)
        if os.path.isfile(self.csv_data):
            os.remove(self.csv_data)
        if os.path.isfile(self.txt_result):
            os.remove(self.txt_result)
        if os.path.isfile(self.json_results):
            os.remove(self.json_results)

    def predict(self, x, alpha=0.05, margin=1, **kwargs):

        self.repetitions = kwargs.pop('repetitions', self.repetitions)
        assert self.repetitions is not None
        if self.repetitions > 0 and self.repetitions < 1. / alpha:
            self.repetitions = int(1.1 / alpha)

        np.savetxt(self.csv_data, x, delimiter=",", fmt='%.8f')
        script_content = """library(\"ecp\")
                            dat = read.csv(\"{}\", header = FALSE)
                            xx <- data.matrix(dat, rownames.force = NA)
                            ret = e.divisive(xx, sig.lvl={}, R={}, k=NULL, min.size={}, alpha=1)
                            library(RJSONIO)
                            jsonfile <- \"{}\"
                            exportJson <- toJSON(ret)
                            write(exportJson, jsonfile)
                            library("rjson")
                            json_data <- fromJSON(file=jsonfile)""".format(self.csv_data, alpha, self.repetitions,
                                                                           margin, self.json_results)

        r_script_handler = open(self.r_script, "w")
        r_script_handler.write(script_content)
        r_script_handler.close()

        import subprocess
        command = 'Rscript {}'.format(self.r_script)
        exit = subprocess.Popen(command.split()).wait()
        self.log.debug('exit status = {}'.format(exit))

        import json
        try:
            with open(self.json_results, "r") as read_file:
                res = json.load(read_file)
        except FileNotFoundError:
            raise EDivisiveRFileNotFoundError('No file {}. Probably something went wrong with the R script'.format(self.json_results))

        if not isinstance(res['p.values'], list):
            res['p.values'] = [res['p.values']]
        self.cp_last_candidate = res['considered.last']
        self.pval_last_candidate = res['p.values'][-1]

        self.cps = res['order.found']
        self.cps.pop(0)
        self.cps.pop(0)

        self.pvals = res['p.values']
        self.pvals.pop()

        self.clear_files()

        return self.cps, self.pvals


def demo(n_jobs=1):
    # cdg.util.logger.set_stdout_level(cdg.util.logger.DEBUG)

    seed = 1234
    np.random.seed(seed)

    import scipy.stats
    delta = 1.
    x0 = scipy.stats.norm.rvs(size=(100, 1))
    x1 = scipy.stats.norm.rvs(size=(200, 1)) + delta
    x = np.concatenate((x0, x1), axis=0)

    margin = 15
    dist_fun = lambda a, b: np.linalg.norm(a-b)
    R=499

    from cdg.changedetection import MeanDistanceScore, Student_t_test
    ttest = Student_t_test()
    mtest = MeanDistanceScore()
    etest = EnergyTest(R=R)
    # rtest = EnergyTest_R(R=R)

    # print('t-test')
    # print(ttest.predict(x0=x0, x1=x1))
    # print(ttest.predicts(x=x, t=100))
    # print('mean test')
    # print(mtest.predict(x0=x0, x1=x1, distance_fun=dist_fun))
    # print(mtest.predicts(x=x, t=100, distance_fun=dist_fun))
    print('energy test')
    print(etest.predict(x0=x0, x1=x1, distance_fun=dist_fun))
    print(etest.predicts(x=x, t=100, distance_fun=dist_fun))
    # print('energy test R')
    # print(rtest.predict(x0=x0, x1=x1 ))
    # print(rtest.predicts(x=x, t=100))

    # print('----- CPM-R')
    # cpmr = CPM(local_test=rtest)
    # print(cpmr.predict(x, margin=margin))
    #
    # print('----- CPM')
    # cpme = CPM(local_test=etest)
    # print(cpme.predict(x, margin=margin, distance_fun=dist_fun))


    # x2 = scipy.stats.norm.rvs(size=(100, 1)) - 2* delta
    # # x = np.concatenate((x, x2), axis=0)
    # x3 = scipy.stats.norm.rvs(size=(40, 1))
    # x = np.concatenate((x, x3), axis=0)


    # print('----- eDiv R')
    # ediv = EDivisive_R()
    # print(ediv.predict(x, margin=margin, R=R))
    # print('failed with:  {} -- {}'.format(ediv.cp_last_candidate,ediv.pval_last_candidate))

    # print('----- Multiple changes')
    # dive = Divisive(local_test=etest)
    # dm = cdg.util.geometry.Eu().distance(x, x)
    # print(dive.predict(dm, margin=margin, R=R, n_jobs=n_jobs, is_dist_mat=True, verbose=True))
    # print('failed with:  {} -- {}'.format(dive.cp_last_candidate,dive.pval_last_candidate))

    import matplotlib.pyplot as plt
    plt.plot(x, '.')
    plt.show()

def test_multidimensional_MultiCPM():

    seed = 1234
    np.random.seed(seed)

    x = scipy.stats.norm.rvs(size=(100, 2))
    x1 = scipy.stats.norm.rvs(size=(200, 2)) + 3
    x = np.concatenate((x, x1), axis=0)
    x2 = scipy.stats.norm.rvs(size=(40, 2))
    x = np.concatenate((x, x2), axis=0)
    x3 = scipy.stats.norm.rvs(size=(60, 2)) + .5
    x = np.concatenate((x, x3), axis=0)


    margin=10
    R=199

    print('----- eDiv R')
    ediv = EDivisive_R()
    print(ediv.predict(x, margin=margin, repetitions=R))
    print('failed with:  {} -- {}'.format(ediv.cp_last_candidate,ediv.pval_last_candidate))


def test_EnergyCPM():

    import scipy.stats
    seed = 1234
    np.random.seed(seed)

    R=199
    delta = .1

    x0 = scipy.stats.norm.rvs(size=(150, 1))
    x1 = scipy.stats.norm.rvs(size=(130, 1)) + delta
    x = np.concatenate((x0, x1))

    import scipy.spatial.distance
    dist_fun = scipy.spatial.distance.euclidean
    dm = scipy.spatial.distance.cdist(x, x, metric='euclidean')

    etest_dm = EnergyTest(repetitions=R)
    cpm_dm = ChangePointMethod(local_test=etest_dm)

    etest_x = EnergyTest(distance_fun=dist_fun)
    cpm_x = EnergyCPM() #CPM(local_test=etest_x)

    import time
    t_0 = time.time()
    r_dm = cpm_dm.predict(x=dm, is_dist_mat=True, margin=2, use_max_stat=False, alpha=0.05)
    t_1 = time.time()
    r_x = cpm_x.predict(x=x, margin=2, alpha=0.05)
    t_2 = time.time()
    print('dm) [{}s] {}'.format(t_1 - t_0, r_dm))
    print('x)[{}s] {}'.format(t_2 - t_1, r_x))


if __name__ == "__main__":
    # demo()
    # test_multidimensional_MultiCPM()
    test_EnergyCPM()
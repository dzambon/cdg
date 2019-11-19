# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements (wrappers of) graph kernels.
# --------------------------------------------------------------------------------
from grakel import GraphKernel
from joblib import Parallel, delayed
from tqdm import tqdm
from cdg.graph import *


class GrakelKernel(Kernel):
    '''
    Interface for the graph kernels provided by `grakel`.
    See https://ysig.github.io/GraKeL/dev/graph_kernel.html#graph-kernel
    '''

    name = 'generic_grakel'
    
    def _kernel(self, source, target, symmetric, paired, verbose):

        if paired:
            raise NotImplementedError()

        if verbose:
            self.log.info('Computing kernel {}...'.format(self.name))

        self.gk.fit(source)

        self.gk.n_jobs = self.n_jobs
        self.gk.verbose = verbose

        ns = len(source)
        nt = len(target)

        window_size = max([400 // ns, 1]) if verbose else nt

        checkpoints = [0]
        n = 1
        while checkpoints[-1] < nt:
            checkpoints.append(n * window_size)
            n += 1

        results = []
        for i in tqdm(range(len(checkpoints)-1), desc=self.name, disable=not verbose):
            results.append(self.gk.transform(target[checkpoints[i]:checkpoints[i+1]]).T)

        return np.hstack(results)
        # if verbose:
        #     self.log.info('Computing kernel {}...'.format(self.name))
        # self.gk.n_jobs = self.n_jobs
        # self.gk.verbose = verbose
        # return self.gk.fit(source).transform(target)

class WeisfeilerLehmanKernel(GrakelKernel):
    name = 'weisfeiler_lehman'

    def __init__(self, verbose=True, normalize=False, n_jobs=1):
        super().__init__()
        self.gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "niter": 5},
                                      {"name": "subtree_wl"}],
                              normalize=normalize, verbose=verbose, n_jobs=n_jobs)
        

class ShortestPathKernel(GrakelKernel):
    name = 'shortest_path'

    def __init__(self, verbose=True, normalize=False, n_jobs=1):
        super().__init__()
        self.gk = GraphKernel(kernel=[{"name": "shortest_path", 'as_attributes': True}],
                              normalize=normalize, verbose=verbose, n_jobs=n_jobs)

    
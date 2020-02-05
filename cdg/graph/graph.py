# --------------------------------------------------------------------------------
# Copyright (c) 2017-2020, Daniele Zambon, All rights reserved.
#
# Define basic classes to deal with graphs.
# --------------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

import cdg.utils

FILENAME_INFO = 'info.txt'
FILENAME_LABELS = 'labels.npy'
FILENAME_ADJACENCY = 'adjacency.npy'
FILENAME_NODE_FEATURE = 'node_feature.npy'
FILENAME_EDGE_FEATURE = 'edge_feature.npy'
FILENAME_DISTANCE_MATRIX = 'distance_matrix.npy'
FILENAME_KERNEL_MATRIX = 'kernel_matrix.npy'

DISTANCE = 'distance'
KERNEL = 'kernel'

def isdataset(path):
    '''
    Check if the folder `path` contains the characteristics files
    associated with an instance of DataSet class.
    :param path: directory to check for the presence of a stored instance of Data Set.
    :return: True/False
    '''
    from os.path import join, isfile
    if not isfile(join(path, FILENAME_LABELS)):
        return False
    elif isfile(join(path, FILENAME_ADJACENCY)):
        return True
    elif isfile(join(path, FILENAME_KERNEL_MATRIX)):
        return True
    elif isfile(join(path, FILENAME_DISTANCE_MATRIX)):
        return True
    else:
        return False

def has_prec_measure(path, measure):
    # todo I don't like this
    from os.path import join, isfile
    if issubclass(type(measure), Distance):
        return isfile(join(path, FILENAME_DISTANCE_MATRIX)), DISTANCE
    elif issubclass(type(measure), Kernel):
        return isfile(join(path, FILENAME_KERNEL_MATRIX)), KERNEL
    else:
        raise ValueError('dataset type not recognised.')

def has_prec_distance(path):
    return has_prec_measure(path=path, measure=Distance())

def has_prec_kernel(path):
    return has_prec_measure(path=path, measure=Distance())

def boost_graph_data_by_vertex_permutations(G, y=None, times=1):
    """
    Boost a data set of graphs by random permuting the vertices of every graphs multiple times.
    :param G: a Graph instance or list of Graph instances.
    :param y: labels associated to graphs G.
    :param times: will produce times*len(G) graphs with permuted vertices.
    """
    if not isinstance(G, list):
        G = [G]
    Gp = []
    yp = []
    for _ in range(times):
        for i, g in enumerate(G):
            p = np.random.permutation(G[0].no_vertices).reshape(-1, 1)
            gp = g.copy()
            gp.permute(perm=p)
            Gp.append(gp)
            if y is not None:
                yp.append(y[i])
    if y is None:
        return Gp
    else:
        return Gp, np.array(yp)


class Graph(object):
    '''
    Graph class.
    A graph is represented in the 'cdg' format, however several conversions are available
    in `cdg.graph.conversion.py`.
    The `cdg` graph format defines a graph of order N as a triple (adj, nf, ef) where
        - adj: np.array(N, N) is an adjacency matrix;
        - nf: np.array(N, Fn) is a node feature matrix, of Fn-dimensional node-attributes;
        - ef: np.array(N, N, Fe) is a edge feature matrix, of Fe-dimensional node-attributes.
    Node and edge features need not to be present.
    '''
    def __init__(self, adj, nf=None, ef=None):

        self.no_vertices = adj.shape[0]
        self.adj = adj

        self.nf = nf if not nf is None else np.zeros((self.no_vertices, 0))
        assert self.nf.ndim == 2
        self.nf_dim = self.nf.shape[-1]

        self.ef = ef if not ef is None else np.zeros((self.no_vertices, self.no_vertices, 0))
        assert self.ef.ndim == 3
        self.ef_dim = self.ef.shape[-1]

        assert self.adj.shape[0] == self.adj.shape[1], 'adjacency matrix is not square'
        assert self.nf.shape[0] == self.no_vertices, 'mismatch in the number of component of nf'
        assert self.ef.shape[0] == self.no_vertices, 'mismatch in the number of component of ef'
    
    def copy(self):
        """Create a copy of the graph (self).""" 
        return Graph(np.copy(self.adj), np.copy(self.nf), np.copy(self.ef))

    def permute(self, perm):
        """
        Permutes the vertices of the graph (self), and rearrange the adj, nf and ef accordingly. The operations overwrite the original graph.
        :param perm: np.array(no_vertices, 1) of integers from 0 to no_vertices-1.
        """
        # assert p.shape[0] == self.no_vertices and p.shape[1] == 1
        self.adj = self.adj[perm, perm.T]
        self.nf = self.nf[perm]
        self.ef = self.ef[perm, perm.T]
        return self
    
    def __str__(self):
        string = "cdg.Graph[no_nodes: {}, nf_dim: {}, ef_dim: {}](".format(self.no_vertices, self.nf_dim, self.ef_dim)
        string += "\n---Adjacency---\n{}".format(self.adj)
        if self.nf_dim > 0:
            string += "\n---NodeFeatures---\n{}".format(self.nf)
        if self.ef_dim > 0:
            string += "\n---EdgeFeatures---"
            for i in range(self.ef_dim):
                string += "\n{}".format(self.ef[:, :, i])
        string += ")"
        return string
    
    
class GraphMeasure(cdg.utils.Loggable, cdg.utils.Pickable):
    '''
    Abstract class that defines the framework for graph measures, like distances and kernels.
    '''
    name = 'generic graph measure'

    def __init__(self, n_jobs=1):
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        self.set_n_jobs(n_jobs=n_jobs)
    
    def set_n_jobs(self, n_jobs):
        self.n_jobs = n_jobs

    def run(self, source, target, symmetric=False, paired=False, verbose=False):
        '''
        Compute the measures between `source` graphs and `target` graphs.
        :param source: list of source graphs
        :param target: list of target graphs
        :param symmetric: flag to reduce (when possible) the computation load; it
            will often compute only the upper triangular matrix and copy the values
            into the lower triangular. `source` and `target` should be of the same.
        :param paired: flag to compute only the measures
            `mea(source[i], target[i]) for i in len(source)`
            the lengths of `source` and `target` must be the smae
        :param verbose: whether to print a status and produce logs.
        :return: (len(source), len(target)) matrix of pairwise measures.
        '''
        ns, nt = len(source), len(target)
        if ns == 0 or nt == 0:
            return np.zeros((ns, nt))
        return self._measure(source, target, symmetric=symmetric, paired=paired, verbose=verbose)
    
    def get_measure_fun(self, **kwargs):
        '''
        :param kwargs: anything necessary for method `run`
        :return: a pure function to compute the graph measure.
        '''
        return lambda s, t: self.run(source=s, target=t, **kwargs)
    
    def _measure(self, source, target, paired, symmetric, verbose):
        '''
        Actual implementation of the measure. Here it is abstract.
        '''
        raise cdg.utils.AbstractMethodError()

    def _handle_single_measure(self, source, target, single_measure_fun, symmetric, paired, verbose):
        from joblib import Parallel, delayed
    
        ns, nt = len(source), len(target)

        # list of positions to compute
        sstt = []
        if paired:
            assert ns != nt
            for i in range(ns):
                sstt.append((i, i))
        else:
            for si in range(ns):
                target_start = si + 1 if symmetric else 0
                for ti in range(target_start, nt):
                    sstt.append((si, ti))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(single_measure_fun)(source[si], target[ti], verbose=verbose)
            for si, ti in tqdm(sstt, desc='{} (n_jobs={})'.format(self.name,self.n_jobs), disable=not verbose))
        
        if paired:
            return np.array(results)
        else:
            result_mat = np.zeros((ns, nt))
            for i in range(len(sstt)):
                si, ti = sstt[i]
                result_mat[si, ti] = results[i]
                if symmetric:
                    result_mat[ti, si] = results[i]
            return result_mat


class Distance(GraphMeasure):
    name = 'generic_distance'
    
    def _measure(self, source, target, symmetric, paired, verbose):
        return self._distance(source=source, target=target, paired=paired, symmetric=symmetric, verbose=verbose)
    
    def _distance(self, source, target, symmetric, paired, verbose):
        raise NotImplementedError()


class Kernel(GraphMeasure):
    name = 'generic_kernel'
    
    def _measure(self, source, target, symmetric, paired, verbose):
        return self._kernel(source=source, target=target, paired=paired, symmetric=symmetric, verbose=verbose)
    
    def _kernel(self, source, target, symmetric, paired, verbose):
        raise NotImplementedError()


class DataSet(cdg.utils.Loggable, cdg.utils.Pickable):
    '''
    Interface class to deal with different data set.
    Think about this a collection of graphs associated with:
        - labels: denoting the class which they belong to
        - distance_measure: a distance function between graphs
        - kernel_measure: a distance function between graphs
    Notice that it is possible not to evaluate the distance/kernel function
    but to exploit precomputed measures stored in a matrix.
    '''
    name = 'Generic'
    notes = None
    is_loaded = True
    
    def __init__(self, graphs=None, labels=None, name=None, store=None, class_to_label=None,
                 distance_measure=None, kernel_measure=None):
        """
        
        :param graphs: (list(cdg.Graph))
        :param labels: (np.ndarray, list(int))
        :param name: (str)
        :param store: (str, def=None) path to where the data set should be stored.
        :param class_to_label: (dict) associate a class name to a integer label.
        :param distance_measure: (np.ndarray, cdg.Distace) either a precomputed matrix of graph distances
            or an instance of cdg.Distance.
        :param kernel_measure: (np.ndarray, cdg.Kernel) either a precomputed matrix of graph distances
            or an instance of cdg.Kernel.
        :param kernel_measure:
        """
        cdg.utils.Pickable.__init__(self)
        cdg.utils.Loggable.__init__(self)
        # self.keep_in_serialising(['name', 'notes', 'no_graphs', 'class_to_label'])
        self.skip_from_serialising(['prec_distance_mat', 'prec_kernel_mat', '_graphs', "distance_measure", "kernel_measure"])
        
        if name is not None:
            self.name = name

        self._graphs = graphs
        self._labels = np.array(labels) if labels is not None else None
        self.no_graphs = len(labels) if graphs is not None else 0

        class_labels_ordered = np.unique(labels)
        self._class_to_label = OrderedDict()
        if class_to_label is not None:
            tmp = class_to_label.copy()
            for c in class_labels_ordered:
                for k, v in class_to_label.items():
                    if c == v:
                        self._class_to_label[k] = v
                        tmp.pop(k)
                        break
            assert len(tmp.keys()) == 0
        else:
            for c in class_labels_ordered:
                self._class_to_label[c] = c
                
        self.elements = OrderedDict()
        for c in self._class_to_label.keys():
            self.elements[c] = np.where(labels == self._class_to_label[c])[0]
        
        if isinstance(distance_measure, np.ndarray):
            self.prec_distance_mat = distance_measure
            self.distance_measure = self._prec_distance_measure
        else:
            self.prec_distance_mat = None
            self.distance_measure = distance_measure
        
        if isinstance(kernel_measure, np.ndarray):
            self.prec_kernel_mat = kernel_measure
            self.kernel_measure = self._prec_kernel_measure
        else:
            self.prec_kernel_mat = None
            self.kernel_measure = kernel_measure
        
        if store is None or store is False:
            pass
        else:
            self.store(path=store)
    
    # @property
    # def classes(self):
    #     return npself.elements.keys()
    def has_prec_distance(self):
        return isinstance(self.prec_distance_mat, np.ndarray)

    def has_prec_kernel(self):
        return isinstance(self.prec_kernel_mat, np.ndarray)

    def get_all_elements(self):
        indices = []
        for c in self.elements.keys():
            indices += list(self.elements[c])
        return np.array(indices)
    
    def get_graphs(self, indices=None, classes=None, format=['idx'], **kwargs):
        if classes is not None:
            assert isinstance(classes, list), 'classes argument should be a list.'
            indices = []
            for c in classes:
                indices += list(self.elements[c])
        
        f_list = format if isinstance(format, list) else [format]
        ret = []
        for f in f_list:
            if f == 'idx':
                ret.append(indices)
            elif f == 'labels':
                ret.append([self._labels[i] for i in indices])
            else:
                gg = [self._graphs[i] for i in indices]
                ret.append(cdg.graph.convert(gg, format_in='cdg', format_out=f, **kwargs))
        
        return ret if isinstance(format, list) else ret[0]
    
    def _prec_distance_measure(self, source, target):
        s = np.array(source) if isinstance(source, list) else source
        t = np.array(target) if isinstance(target, list) else target
        return self.prec_distance_mat[s][:, t]
    
    def _prec_kernel_measure(self, source, target):
        s = np.array(source) if isinstance(source, list) else source
        t = np.array(target) if isinstance(target, list) else target
        return self.prec_kernel_mat[s][:, t]
    
    def store(self, path, skip_graphs=False, notes=None,
              dist_mat=None, dist_gen=None, dist_notes=None,
              kernel_mat=None, kernel_gen=None, kernel_notes=None):
        
        import os
        assert os.path.isdir(path), 'The requested path <{}> does not exist.'.format(path)
        
        if path[-1] == '/':
            path = path[:-1]
        
        info_file = open('{}/{}'.format(path, FILENAME_INFO), 'w')
        info_file.write('Graph data set generated from class {}\n'.format(DataSet))
        info_file.write(' - name: {}\n'.format(self.name))
        info_file.write(' - number of graphs: {}\n'.format(self.no_graphs))
        info_file.write(' - classes: {}\n'.format(self.elements.keys()))
        info_file.close()
        
        DataSet.store_labels(path=path, labels=self._labels, class_to_label=self._class_to_label)
        
        if not skip_graphs:
            adj = []
            nf = []
            ef = []
            for g in self._graphs:
                adj.append(g.adj[None, ...])
                nf.append(g.nf[None, ...])
                ef.append(g.ef[None, ...])
            DataSet.store_npy_graphs(path=path, adj=np.vstack(adj), nf=np.vstack(nf), ef=np.vstack(ef))
        
        if dist_mat is True:
            dist_mat = self.prec_distance_mat
        if dist_mat is not None:
            DataSet.store_distance(path=path, generated=dist_gen,
                                   distance_matrix=dist_mat, measure_notes=dist_notes, notes=notes)
        
        if kernel_mat is True:
            kernel_mat = self.prec_kernel_mat
        if kernel_mat is not None:
            DataSet.store_kernel(path=path, generated=kernel_gen,
                                 kernel_matrix=kernel_mat, measure_notes=kernel_notes, notes=notes)
        
        return self
    
    @staticmethod
    def store_npy_graphs(path, adj, nf, ef):
        np.save('{}/{}'.format(path, FILENAME_ADJACENCY), adj)
        np.save('{}/{}'.format(path, FILENAME_NODE_FEATURE), nf)
        np.save('{}/{}'.format(path, FILENAME_EDGE_FEATURE), ef)
    
    @staticmethod
    def store_labels(path, labels, class_to_label):
        np.save('{}/{}'.format(path, FILENAME_LABELS), {'labels': labels, 'class_to_label': class_to_label})
    
    @staticmethod
    def store_distance(path, distance_matrix, generated, measure_notes, notes):
        np.save('{}/{}'.format(path, FILENAME_DISTANCE_MATRIX),
                {'dist_mat': distance_matrix,
                 'generated': generated,
                 'distance_measure': measure_notes,
                 'notes': notes})
    
    @staticmethod
    def store_kernel(path, kernel_matrix, generated, measure_notes, notes):
        np.save('{}/{}'.format(path, FILENAME_KERNEL_MATRIX),
                {'kernel_mat': kernel_matrix,
                 'generated': generated,
                 'distance_measure': measure_notes,
                 'notes': notes})
    
    @staticmethod
    def load_dataset(path, name=None,
                     precomputed_distance=False, distance_measure=None,
                     precomputed_kernel=False, kernel_measure=None,
                     skip_graphs=False):
        if path[-1] == '/':
            path = path[:-1]
        
        labels, class_to_label = DataSet.load_labels(path)
        gg = DataSet.load_graphs(path) if not skip_graphs else np.arange(0, labels.shape[0])
        dist_mea = DataSet.load_prec_distance(path) if precomputed_distance else distance_measure
        kernel_mea = DataSet.load_prec_kernel(path) if precomputed_kernel else kernel_measure
        dataset = DataSet(graphs=gg, labels=labels, distance_measure=dist_mea, kernel_measure=kernel_mea,
                          class_to_label=class_to_label)
        if name is not None:
            dataset.name = name
        return dataset
    
    @staticmethod
    def load_labels(path):
        lab_dic = np.load('{}/{}'.format(path, FILENAME_LABELS), allow_pickle=True).item()
        ctl = lab_dic['class_to_label'] if 'class_to_label' in lab_dic.keys() else None
        return lab_dic['labels'], ctl
    
    @staticmethod
    def load_graphs(path):
        adj = np.load('{}/{}'.format(path, FILENAME_ADJACENCY), allow_pickle=True)
        nf = np.load('{}/{}'.format(path, FILENAME_NODE_FEATURE), allow_pickle=True)
        ef = np.load('{}/{}'.format(path, FILENAME_EDGE_FEATURE), allow_pickle=True)
        gg = []
        for i in range(adj.shape[0]):
            gg.append(Graph(adj=adj[i], nf=nf[i], ef=ef[i]))
        return gg
    
    @staticmethod
    def load_prec_distance(path):
        dict = np.load('{}/{}'.format(path, FILENAME_DISTANCE_MATRIX), allow_pickle=True).item()
        cdg.utils.logger.info('using distance matrix:\n\t* generated: {}\n\t* notes: {}'.format(dict['generated'],
                                                                                                dict['notes']))
        return dict['dist_mat']
    
    @staticmethod
    def load_prec_kernel(path):
        dict = np.load('{}/{}'.format(path, FILENAME_KERNEL_MATRIX), allow_pickle=True).item()
        cdg.utils.logger.info('using kernel matrix:\n\t* generated: {}\n\t* notes: {}'.format(dict['generated'],
                                                                                              dict['notes']))
        return dict['kernel_mat']

    @staticmethod
    def convert_cdg2dissmat(dataset_path, name, cls2lab=None, notes=None):
        import os
        import pickle

        if notes is None:
            notes = 'Generated with GMT (VJ method).'

        assert os.path.isdir(dataset_path), 'Directory {} not found.'.format(dataset_path)
        dissmat_filepath = os.path.join(dataset_path, 'dissimilarity_matrix.pkl')
        assert os.path.isfile(dissmat_filepath), 'File {} not found.'.format(dissmat_filepath)
        dissmat_file = open(dissmat_filepath, 'rb')
        dissmat_dict = pickle.load(dissmat_file)
        dissmat_file.close()
        
        dissmat = dissmat_dict['dissMat']
        generation_date = dissmat_dict['generated']
        
        map_filepath = os.path.join(dataset_path, 'graph_name.map')
        assert os.path.isfile(map_filepath), 'File {} not found.'.format(dissmat_filepath)
        map_file = open(map_filepath, 'rb')
        idx, lab = [], []
        _ = map_file.readline()  # generated
        _ = map_file.readline()  # header
        for line in map_file.readlines():
            sp = line.decode("utf-8").split(',')
            idx.append(int(sp[0]))
            lab.append(int(sp[2]))
        map_file.close()
        
        dataset = DataSet(graphs=idx, labels=np.array(lab), name=name, class_to_label=cls2lab,
                          distance_measure=dissmat)
        
        dataset.store(path=dataset_path, skip_graphs=True,
                      dist_mat=True, dist_gen=generation_date, dist_notes=notes)

        return dataset

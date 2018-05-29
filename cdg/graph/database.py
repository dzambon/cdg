# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# This is intended as a python interface for dealing with the IAM Graph Database
# repository [1], Delaunay Graphs [ssci17] and for interfacing with the
# GraphMatchingToolkit [2].
#
#
# References:
# ---------
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
# [ssci17] 
#   Zambon, Daniele, Livi, Lorenzo and Alippi, Cesare. 
#   Detecting Changes in Sequences of Attributed Graphs.
#   IEEE Symposium Series in Computational Intelligence (2017).
#
# [1] Riesen, Kaspar, and Horst Bunke. "IAM graph database repository for graph 
#   based pattern recognition and machine learning." Structural, Syntactic, and 
#   Statistical Pattern Recognition (2008): 287-297.
#
# [2] 
#   K. Riesen, S. Emmenegger and H. Bunke. 
#   A Novel Software Toolkit for Graph Edit Distance Computation. 
#   In W.G. Kropatsch et al., editors, Proc. 9th Int. Workshop on Graph Based 
#   Representations in Pattern Recognition, LNCS 7877, 142–151, 2013.
#
#
# ------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# ------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 16/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import xml.etree.ElementTree as et
import numpy as np
import scipy.spatial
import subprocess
import datetime
import pickle
import os
import sys
import joblib
from tqdm import tqdm
import cdg.graph.graph
import cdg.util.logger
import cdg.util.serialise
import cdg.graph.dissimilarity
import cdg.util.errors

def dot2gxl(folder):
    ct = 0
    for filedot in os.listdir(folder):
        if filedot.endswith(".dot"):
            filegxl = filedot[:-4] + ".gxl"
            command = "dot2gxl -g " + folder + "/" + filedot
            cdg.util.logger.glog().debug("executing: " + command)
            # out=subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()
            out = subprocess.Popen(command.split(), stdout=open(folder + "/" + filegxl, 'wb')).wait()
            cdg.util.logger.glog().debug(out)
            ct += 1
    return ct


class Database(cdg.util.logger.Loggable, cdg.util.serialise.Pickable):
    '''
    Interface class to deal with different database.
    '''

    # flag: whether or not to precompute the entire dissimilarity matrix.
    _precomputed_dissimilarity_matrix = False
    # filename for the graph-name map
    _filename_graph_name_map = "graph_name.map"
    # filename for pickle storing the dissimilarity matrix
    _pickle_dissimilarity_matrix = "dissimilarity_matrix.pkl"
    # flag to check whether the dissimilarity matrix is loaded
    _dissimilarity_matrix_loaded = False

    name = 'Generic'
    notes = None

    def __init__(self, path, dissimilarity_instance=None):
        """
        :param path: of the dataset
        :param dissimilarity_instance: instance of a subclass of cdg.graph.Dissimilarity
        """
        cdg.util.logger.Loggable.__init__(self)
        cdg.util.serialise.Pickable.__init__(self)

        self.keep_in_serialising(
            ['path', 'name', 'notes', 'no_graphs', 'classes'])

        self.path = path
        if self.path.endswith('/'):
            self.path = self.path[:-1]
        self.graph_name_and_class = []
        self.classes = []
        self.elements = {}

        self.set_dissimilarity_instance(dissimilarity_instance)

        if not os.path.isdir(self.path):
            self.log.warning(self.path + ' does not exist')
        else:
            foundGxlorNpy = False
            folders = [self.path]
            for folder in folders:
                for fname in os.listdir(folder):
                    if fname.endswith('.gxl') or fname.endswith('.npy'):
                        foundGxlorNpy = True
                        break
                    if os.path.isdir(folder + '/' + fname):
                        folders.append(folder + '/' + fname)
                if foundGxlorNpy:
                    break

            if not foundGxlorNpy and os.path.isdir(self.path):
                self.log.debug('no .gxl or .npy in path: ' + self.path)

    def set_dissimilarity_instance(self, dissimilarity_instance):
        self.dissimilarity_instance = dissimilarity_instance
        if dissimilarity_instance is not None:
            if not issubclass(type(dissimilarity_instance), cdg.graph.dissimilarity.Dissimilarity):
                raise ValueError(' argument dissimilarity_instance has to be an instance of a '
                                 'subclass of cdg.graph.Dissimilarity')
            else:
                self.dissimilarity_instance.set_properties(self.get_GMT_properties())

    def __str__(self):
        string = ''
        string += "{} = {}\n".format('name', self.name)
        string += "{} = {}\n".format('path', self.path)
        string += "{} = {}\n".format('classes', self.classes)
        string += "{} = {}\n".format('notes', self.notes)
        return string

    def load_graph_name_map(self):
        """
        Loads the graph-name maps. It is a structure storing the filename of the .gxl file,
        a unique integer identifier and the class to which it belongs.
        The python structure is dictionary `elements` of the form
            {'0': [0, 1, ... ],
               ...
             '11': [ ... 2998, 2999],
             '12': [3000, 3001, ...],
               ...
            }
        The graph-map file is a .csv representing the following table:
            | incremental ID | filename of the graph | class |
            --------------------------------------------------
            | 0              | 0/del_0_0.gxl         | 0     |
            | 1              | 0/del_0_1.gxl         | 0     |
                ...
            | 2998           | 11/del_11_998.gxl     | 11    |
            | 2999           | 11/del_11_999.gxl     | 11    |
            | 3000           | 12/del_12_0.gxl       | 12    |
            | 3001           | 12/del_12_1.gxl       | 12    |
                ...

        """
        ct = 0

        if not os.path.isfile(self.path + "/" + self._filename_graph_name_map):
            self._generate_graph_name_map()

        self.graph_name_and_class = []
        with open(self.path + "/" + self._filename_graph_name_map, 'r') as f:
            for line in f:
                if line[0] != '#':
                    tokens = line.split(',')
                    self.graph_name_and_class.append((int(tokens[0]), tokens[1], tokens[2]))
                    ct += 1
        self.no_graphs = ct

        self.classes = []
        self.elements = {}
        for g in self.graph_name_and_class:
            found = False
            for c in self.classes:
                if g[2] == c:
                    found = True
                    self.elements[c].append(int(g[0]))
                    break
            if not found:
                self.classes.append(g[2])
                self.elements[g[2]] = [g[0]]

    def _generate_graph_name_map(self):
        # numGraphs = _find_class(self.path)
        graphNameList = []
        for file in os.listdir(self.path):
            if file.endswith(".gxl"):
                graphNameList.append(file)
        graphNameList.sort()

        self.graph_name_and_class = []
        ct = 0
        for graph in tqdm(graphNameList, desc='generating graph-name map'):
            type = Database._find_class(self.path, graph)
            self.graph_name_and_class.append((ct, graph, type))
            ct += 1
        self._save_graph_name_map()

    def get_all_elements(self):
        el = []
        for k in self.elements.keys():
            el += self.elements[k]
        return el

    @staticmethod
    def _find_class(db_path, name=None):

        files = [db_path + "/train.cxl", db_path + "/test.cxl",
                 db_path + "/valid.cxl", db_path + "/validation.cxl"]
        for file in files:
            try:
                tree = et.parse(file)
                entries = tree.findall(".//print")
                for e in entries:
                    if e.attrib['file'] == name:
                        return e.attrib['class']
            except FileNotFoundError:
                tmp = 0

        raise ValueError("class of %s not found" % name)

    def _save_graph_name_map(self):
        f = open(self.path + "/" + self._filename_graph_name_map, 'w')
        f.write("# Generated: %s" % datetime.datetime.now().strftime('%G/%m/%d %H:%M'))
        f.write("\n# Header description: incremental ID, filename of the graph, class,")
        ct = 0
        for graphEntry in self.graph_name_and_class:
            f.write("\n%d,%s,%s," % graphEntry)
            ct += 1

        self.no_graphs = ct
        f.close()

    def load_dissimilarity_matrix(self):
        """ Loads the dissimilarity matrix. """
        if self._dissimilarity_matrix_loaded:
            return
        self.load_graph_name_map()
        if os.path.isfile(self.path + "/" + self._pickle_dissimilarity_matrix):
            # load from pickle file.
            dissMatAndInfo = pickle.load(
                open(self.path + "/" + self._pickle_dissimilarity_matrix, "rb"))
            self.dissimilarity_matrix = dissMatAndInfo['dissMat']

            if self._precomputed_dissimilarity_matrix \
                    and np.min(self.dissimilarity_matrix) < 0:
                raise cdg.util.erros.CDGForbidden(
                    "dz: you are trying to load the dissimilarity matrix as"
                    + " precomputed, but it is not fully completed yet.")
        else:
            self._generate_dissimilarity_matrix()

        self._dissimilarity_matrix_loaded = True

    def _generate_dissimilarity_matrix(self):

        if self._precomputed_dissimilarity_matrix:
            self._precompute_split()
        else:
            sys.stdout.write('init dissimilarity matrix...')
            empty_dissmat = -np.ones((len(self.graph_name_and_class),
                                      len(self.graph_name_and_class)),
                                     dtype=np.float32)
            self._save_dissimilarity_matrix(dissimilarity_matrix=empty_dissmat)
            print(' done')

    def _save_dissimilarity_matrix(self, dissimilarity_matrix=None):
        if dissimilarity_matrix is not None:
            self.dissimilarity_matrix = dissimilarity_matrix.copy()
            no_graphs = dissimilarity_matrix.shape[0]
            if no_graphs != dissimilarity_matrix.shape[1]:
                print(dissimilarity_matrix.shape)
                raise ValueError('Something wrong with the matrix')

        no_graphs = self.dissimilarity_matrix.shape[1]
        dissMatAndInfo = {'dissMat': self.dissimilarity_matrix,
                          'generated': datetime.datetime.now().strftime(
                              '%G/%m/%d %H:%M')}

        # Save as pickle file.
        pickle.dump(dissMatAndInfo,
                    open(self.path + "/" + self._pickle_dissimilarity_matrix,
                         "wb"))

    def get_name_and_class(self, id):
        for el in self.graph_name_and_class:
            if el[0] == id:
                return el
        return None

    def get_sub_dissimilarity_matrix(self, rows, columns, rows_at_a_time=None, n_jobs=1):
        """
        Retrieve ---in case, compute---
        :param rows: a list of graph identifiers in the graph-name map.
        :param columns: a list of graph identifiers in the graph-name map.
        :param rows_at_a_time: ... not implemented yet
        :param n_jobs: jobs in joblib. Default is 1.
        :return:
        """

        self.load_dissimilarity_matrix()

        # parse input
        if rows_at_a_time is not None:
            raise NotImplementedError()
        if type(rows) is int:
            rows = [rows]
        if type(columns) is int:
            columns = [columns]

        # sets of columns and rows
        rows_set = np.array(list(set(rows)))
        columns_set = np.array(list(set(columns)))

        # if not already computed
        if not self._precomputed_dissimilarity_matrix:

            # select what to compute
            to_be_computed = np.full((rows_set.size, columns_set.size), False)
            for i in range(0, len(rows_set)):
                j_indices = np.where(
                    self.dissimilarity_matrix[rows_set[i], columns_set] < 0)
                to_be_computed[i, j_indices] = True

            if np.any(to_be_computed):
                # compute dissimilarities
                try:
                    fun = type(self.dissimilarity_instance).static_run
                except AttributeError as e:
                    raise AttributeError(str(e) + '. Probably you haven\'t provided a '
                                                  'dissimilarity instance')
                gnac = self.graph_name_and_class
                row_list = []
                cols_list = []
                for i in range(0, len(rows_set)):
                    row_list.append([rows_set[i]])
                    cols_list.append(columns_set[np.where(to_be_computed[i, :])].tolist())
                # run
                self.log.info('computation starts now...')
                if n_jobs != 1:
                    dis_mat_list = joblib.Parallel(n_jobs=n_jobs, verbose=5) \
                        (joblib.delayed(fun)(instance=self.dissimilarity_instance,
                                             source=row_list[i],
                                             target=cols_list[i],
                                             # path=self.path,
                                             graph_name_and_class=gnac,
                                             verbose=True,
                                             id=str(i),
                                             ) for i in range(len(rows_set)))
                else:
                    dis_mat_list = []
                    for i in range(0, len(rows_set)):
                        dis_mat_list.append(
                            fun(instance=self.dissimilarity_instance,
                                source=row_list[i],
                                target=cols_list[i],
                                # path=self.path,
                                graph_name_and_class=gnac,
                                verbose=True,
                                id=str(i),
                                ))
                self.log.info('computation starts now... done')
                # store dissimilarities
                for i in range(0, len(rows_set)):
                    self.dissimilarity_matrix[rows_set[i], cols_list[i]] = dis_mat_list[i]
                # save computed matrix
                self._save_dissimilarity_matrix()

        # extract sub matrix
        sub_matrix = np.zeros((len(rows), len(columns)))
        for i in range(0, len(rows)):
            sub_matrix[i, :] = self.dissimilarity_matrix[rows[i], columns]
        return sub_matrix

    def generate_bootstrapped_stream(self, classes, length, prc=None):
        """

        :param classes:
        :param length:
        :param prc:
        :return:
        """

        if prc is None:

            bin = []  # [1./len(classes) for c in classes]
            for i in range(0, len(classes)):
                bin.append((i + 1.) / len(classes))

            binStream = []
            for t in range(0, length):
                binStream.append(bin)

        elif len(prc) == len(classes):

            bin = [prc[0]]  # [1./len(classes) for c in classes]
            for i in range(1, len(prc)):
                bin.append(bin[-1] + prc[i])
            if bin[-1] < 0.99:
                raise ValueError("prc do not sum to 1")
            else:
                bin[-1] = 1

            binStream = []
            for t in range(0, length):
                binStream.append(bin)

        elif len(prc) == length:
            binStream = []
            for prc_t in prc:
                bin = [prc_t[0]]  # [1./len(classes) for c in classes]
                for i in range(1, len(prc_t)):
                    bin.append(bin[-1] + prc_t[i])
                if bin[-1] < 0.99:
                    raise ValueError("prc do not sum to 1")
                else:
                    bin[-1] = 1

                binStream.append(bin)

        else:
            raise cdg.util.errors.CDGError("length mismatch")

        if type(prc) is list and len(prc) == length:
            self.log.debug('bootstrapping with drift (probably)...')
        else:
            self.log.debug('bootstrapping (prc = %s)...' % prc)

        elements = []
        for c in classes:
            elements += self.elements[c]

        stream_tmp = []
        for i in range(0, len(classes)):
            stream_tmp.append(np.random.choice(elements, length, True))

        r = np.random.rand(length)
        cc = []
        for t in range(0, length):
            i = 0
            while r[t] > binStream[t][i]:
                i += 1
            cc.append(i)

        stream = [stream_tmp[cc[t]][t] for t in range(0, length)]

        return stream


class IAMGeometric(Database):
    name = 'Geometric'

    def get_GMT_properties(self):
        propertyDict = {}

        propertyDict['numOfNodeAttr'] = 2
        propertyDict['nodeAttr0'] = 'x'
        propertyDict['nodeAttr1'] = 'y'

        propertyDict['nodeCostType0'] = 'squared'
        propertyDict['nodeCostType1'] = 'squared'

        propertyDict['nodeAttr0Importance'] = 1.0
        propertyDict['nodeAttr1Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.5

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict


class IAMMolecule(Database):
    name = 'Molecule'

    def get_GMT_properties(self):
        propertyDict = {}

        propertyDict['numOfNodeAttr'] = 1
        propertyDict['nodeAttr0'] = 'chem'

        propertyDict['nodeCostType0'] = 'sed'
        propertyDict['nodeAttr0Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 1
        propertyDict['edgeAttr0'] = 'valence'
        propertyDict['edgeCostType0'] = 'squared'
        propertyDict['edgeAttr0Importance'] = 1.0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.5

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict


class Letter(IAMGeometric):
    name = 'Letter'

    def __init__(self, path, distortion, dissimilarity_instance):
        IAMGeometric.__init__(self, path=path + "/" + distortion,
                              dissimilarity_instance=dissimilarity_instance)
        self.main_path = path
        self.distortion = distortion

        self.notes = 'distortion = ' + str(distortion)


class Mutagenicity(IAMMolecule):
    name = 'Mutagenicity'


class AIDS(IAMMolecule):
    name = 'AIDS'

    def get_GMT_properties(self):
        propertyDict = IAMMolecule.get_GMT_properties(self)
        propertyDict['nodeAttr0'] = 'symbol'

        return propertyDict


class VectorWeighted(Database):
    """
    Graphs with vertex and edge attribute from R^n.
    """

    name = 'VectorWeighted'

    def get_GMT_properties(self):

        propertyDict = {}

        propertyDict['node'] = 1.0
        propertyDict['edge'] = 0.1

        propertyDict['numOfNodeAttr'] = 1
        propertyDict['nodeAttr0'] = 'weight'
        propertyDict['nodeCostType0'] = 'csvDouble'
        propertyDict['nodeAttr0Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 1
        propertyDict['edgeAttr0'] = 'weight'
        propertyDict['edgeCostType0'] = 'csvDouble'
        propertyDict['edgeAttr0Importance'] = 1.0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 2

        propertyDict['alpha'] = 1

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict



class Delaunay(VectorWeighted):
    """

    The Delaunay graph dataset adopt an attribute format as comma-separated 
    double values which is not supported by `GraphMatchingToolkit` [1] by default.
    You have three options:
       1. (Lazy solution) Don't used the Delaunay dataset; 
       2. Edit the `database.Delaunay` class to export `.gxl` file in a format 
          compliant to the `database.Letter` dataset.
       3. (Recommended) get the edited version of the `GraphMatchingToolkit` 
          available [here](https://github.com/dan-zam/graph-matching-toolkit).
    """

    name = 'Delaunay'

    def generate_new_dataset(self, seed_points=10, classes=20, no_graphs=10, format=['gxl']):
        """
        
        :param x: (no_train, d) training data.
        :param seed_points: If `np.array` (no_points, 2) seed points for the graph generating mechanism. 
            If `int` then no_points=seed_points points are created (def = 10)
        :param classes: If `list` then it is a list of class identifiers. If `int` then all classes from 0
            to `classes` are created. Class identifiers are nonnegative integers: `id = 0, 1, 2, ...` are 
            all admissible classes. Class 0 is usually intended as reference class. As `id` increases, 
            class `id` get 'closer' to class 0. (def = 20)
        :param no_graphs: number of graphs to be generated. If `int` every class will have the same
            number of graphs, otherwise it can be a dictionary {classID: no graphs} (def = 10)
        :param format: `['gxl', 'npy']` formats for saving the generated graphs. 
            'gxl' : each graph is a independent .gxl file
            'npy' : each class is stored in two .npy files: `np.array` of points (no_graphs, no_points, 2) 
                and `np.array` of adjacency matrices (no_graphs, no_points, no_points).
        """

        # parse seed_points
        radius = 10.
        if type(seed_points) is int:
            seed_points = np.random.rand(seed_points, 2) * radius
            no_points = seed_points.shape[0]

        # parse classes
        if isinstance(classes, list):
            self.classes = classes.copy()
        elif isinstance(classes, int):
            self.classes = [i for i in range(classes + 1)]

        # parse no_graphs
        if isinstance(no_graphs, int):
            no_graphs_dict = {c: no_graphs for c in self.classes}
        else:
            no_graphs_dict = no_graphs

        if os.path.isfile(self.path + "/" + self._filename_graph_name_map):
            os.remove(self.path + "/" + self._filename_graph_name_map)
        if os.path.isfile(self.path + "/" + self._pickle_dissimilarity_matrix):
            os.remove(self.path + "/" + self._pickle_dissimilarity_matrix)

        for c in self.classes:
            if c == 0:
                points = seed_points
            else:
                points = seed_points.copy()
                for i in range(no_points):
                    points[i, 0] += np.sin(np.random.rand(1) * 2. * np.pi) * radius
                    points[i, 1] += np.cos(np.random.rand(1) * 2. * np.pi) * radius
                radius *= .66

            del_gen = DelaunayGenerator(path=self.path + '/' + str(c), classID=c)
            del_gen.set_fundamental_points(points=points)
            sigma = 1.

            adjs_list = []
            points_list = []
            for i in range(no_graphs_dict[c]):
                new_points, new_adjmat = del_gen.generate_new_graph(sigma=sigma)
                if 'gxl' in format:
                    del_gen.save_GXL(new_points, new_adjmat, del_gen.nameFamily + '_' + str(i))
                if 'npy' in format:
                    points_list.append(new_points[None, ...])
                    adjs_list.append(new_adjmat[None, ...])
            if 'npy' in format:
                np.save(del_gen.path + '/points_batch.npy', np.vstack(points_list))
                np.save(del_gen.path + '/adjacency_batch.npy', np.vstack(adjs_list))

    def _generate_graph_name_map(self):

        graphNameList = []
        # for file in os.listdir(self.path+"/Training")+os.listdir(self.path+"/Validation")+os.listdir(self.path+"/Test"):
        for pathAbs, s, filesAbs in os.walk(self.path + '/'):
            for fileAbs in tqdm(filesAbs):
                if fileAbs.endswith(".gxl"):
                    pathRel = os.path.relpath(pathAbs, self.path)
                    fileRel = os.path.join(pathRel, fileAbs)
                    # print(fileRel)
                    graphNameList.append(fileRel)

        graphNameList.sort()

        self.graph_name_and_class = []
        ct = 0
        for graph in graphNameList:
            className = graph.split("_")[1]
            self.graph_name_and_class.append((ct, graph, className))
            ct += 1
            self.log.debug((ct, graph, className))

        self._save_graph_name_map()

    def get_GMT_properties(self):
        propertyDict = super().get_GMT_properties()
        propertyDict['numOfEdgeAttr'] = 0
        # propertyDict['edgeAttr0'] = 'weight'
        # propertyDict['edgeCostType0'] = 'csvDouble'
        # propertyDict['edgeAttr0Importance'] = 1.0
        propertyDict['alpha'] = 0.7
        return propertyDict


class Markov(VectorWeighted):
    name = 'Markov'

    def __init__(self, path, difficulty, dissimilarity_instance, dot_to_gxl=False):
        """

        :param ...
        :param dot_to_gxl: natively the dataset is in dot format, if this flag is set to True, then
            it automatically generates gxl files for the dot ones.
        """
        VectorWeighted.__init__(self, path=path + "/" + str(difficulty),
                              dissimilarity_instance=dissimilarity_instance)
        self.main_path = path
        self.difficulty = difficulty

        self.name = 'Markov_' + str(difficulty)
        self.notes = 'difficulty = ' + str(difficulty)

        if dot_to_gxl:
            for pathAbs, s, filesAbs in os.walk(self.path):
                found_dot = False
                found_gxl = False
                for fileAbs in filesAbs:
                    if fileAbs.endswith(".dot"):
                        found_dot = True
                    elif fileAbs.endswith(".gxl"):
                        found_gxl = True
                        break
                if found_dot and not found_gxl:
                    dot2gxl(pathAbs)

    def _generate_graph_name_map(self):

        # numGraphs = find_class(self.path)
        graphNameList = []
        # for file in os.listdir(self.path+"/Training")+os.listdir(self.path+"/Validation")+os.listdir(self.path+"/Test"):
        for pathAbs, s, filesAbs in os.walk(self.path):
            for fileAbs in filesAbs:
                if fileAbs.endswith(".gxl"):
                    pathRel = os.path.relpath(pathAbs, self.path)
                    fileRel = os.path.join(pathRel, fileAbs)
                    self.log.info(fileRel)
                    graphNameList.append(fileRel)
        graphNameList.sort()

        self.graph_name_and_class = []
        ct = 0
        for graph in graphNameList:
            className = graph.split("_")[2][0]
            self.graph_name_and_class.append((ct, graph, className))
            ct += 1
            self.log.debug((ct, graph, className))

        self._save_graph_name_map()

    def get_GMT_properties(self):
        propertyDict = super().get_GMT_properties()
        propertyDict['node'] = 2.0
        propertyDict['edge'] = 0.1
        propertyDict['alpha'] = 0.9
        return propertyDict

    def generate_GXL(self):
        ct = 0
        ct += dot2gxl(self.path + "/Training")
        ct += dot2gxl(self.path + "/Validation")
        ct += dot2gxl(self.path + "/Test")
        print("created " + str(ct) + " .gxl files in " + self.path)




class KaggleSeizure(Database):
    # Flag for adopting spektral for loading data. Otherwise, a data structure compliant
    # with cdg is necessary.
    _spektral_handler = False

    name = 'KaggleSeizure'

    def __init__(self, path, dissimilarity_instance=None,
                 spektral_handler=True, spektral_id=None, spektral_dataset='detection',
                 spektral_gt='corr'):
        """

        :param path: of the dataset
        :param dissimilarity_instance: instance of a subclass of cdg.graph.Dissimilarity
        :param spektral_handler: Flag to use spektral module
        :param spektral_id: name of the kaggle dataset according to spektral notation
        :param spektral_dataset: ['prediction', 'detection']
        :param spektral_gt: ['corr', 'dpli']
        """
        super().__init__(path=path, dissimilarity_instance=dissimilarity_instance)
        if not os.path.isdir(self.path):
            raise FileNotFoundError('cdg: Folder {} does not exist, yet. '
                                    'Either create it or change folder'.format(self.path))

        self.spektral_name = spektral_id
        self.spektral_dataset = spektral_dataset
        self.spektral_gt = spektral_gt
        self.name = 'KaggleSeizure'
        self.notes = None

        self._spektral_handler = spektral_handler

    def _generate_graph_name_map(self):

        if not self._spektral_handler:
            return super()._generate_graph_name_map()

        # Import data with spectral
        import spektral.datasets.kaggle_seizure
        data = spektral.datasets.kaggle_seizure.load_data(patients=self.spektral_name,
                                                          dataset=self.spektral_dataset,
                                                          graph_type=self.spektral_gt)
        # Unpack data after loading
        _, _, _, class_labels, _, _, _ = data

        self.graph_name_and_class = []
        ct = 0
        for id in tqdm(range(len(class_labels)), desc='generating graph-name map'):
            self.graph_name_and_class.append((id, 'n.d.', int(class_labels[id])))
        self._save_graph_name_map()

    @staticmethod
    def _compute_submatrix(rows, columns, id, gmt, graph_name_and_class, path):
        raise NotImplementedError()

    def get_GMT_properties(self):
        propertyDict = {}

        propertyDict['numOfNodeAttr'] = 2
        propertyDict['nodeAttr0'] = 'x'
        propertyDict['nodeAttr1'] = 'y'

        propertyDict['nodeCostType0'] = 'squared'
        propertyDict['nodeCostType1'] = 'squared'

        propertyDict['nodeAttr0Importance'] = 1.0
        propertyDict['nodeAttr1Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.5

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict


class KaggleSeizureDog(KaggleSeizure):
    def __init__(self, path, patient, dissimilarity_instance=None, spektral_handler=True):
        if spektral_handler:
            spektral_id = 'Dog_{}_FC'.format(patient)
        else:
            spektral_id = None
        super().__init__(path=path, dissimilarity_instance=dissimilarity_instance,
                         spektral_handler=spektral_handler, spektral_id=spektral_id)
        self.name = 'KaggleDog_{}'.format(patient)


class KaggleSeizureHuman(KaggleSeizure):
    def __init__(self, path, patient, dissimilarity_instance=None, spektral_handler=True):
        if spektral_handler:
            spektral_id = 'Patient_{}_FC'.format(patient)
        else:
            spektral_id = None
        super().__init__(path=path, dissimilarity_instance=dissimilarity_instance,
                         spektral_handler=spektral_handler, spektral_id=spektral_id)
        self.name = 'KaggleHuman_{}'.format(patient)


class DelaunayGenerator(cdg.util.logger.Loggable):
    def __init__(self, path='.', classID=42, nameFamily=None):
        self.classID = classID
        self.path = path
        if nameFamily is None:
            self.nameFamily = 'del_' + str(classID)
        else:
            self.nameFamily = nameFamily
        if not os.path.exists(self.path):
            self.log.warning(
                "Path %s does not exists already. It's about to be created." % self.path)
            os.makedirs(self.path)

    def set_fundamental_points(self, points):
        self.points = points
        if points.shape[1] != 2:
            raise cdg.util.errors.CDGError("dimension must be two")
        self.noVertices = points.shape[0]

    def generate_fundamental_points(self, noVertices):
        self.setFundamentalPoints(self, np.random.randn(noVertices, 2))

    def generate_new_graph(self, sigma=.2, radius=1):
        newPoints = self.points + np.random.randn(self.noVertices, 2) * sigma
        newPoints *= radius
        newAdjMat = DelaunayGenerator.generate_delaunay_adjacency(newPoints)

        return newPoints, newAdjMat

    def save_GXL(self, newPoints, newAdjMat, nameGraph):

        f = open(self.path + '/' + nameGraph + '.gxl', 'w')

        f.writelines('<?xml version="1.0" encoding="UTF-8"?>\n')
        # f.writelines('<!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd">\n')
        # f.writelines('<gxl xmlns:xlink=" http://www.w3.org/1999/xlink">\n')
        f.writelines('<gxl>\n')
        f.writelines(
            '\t<graph id="%s" edgeids="false" edgemode="undirected">\n' % nameGraph)
        f.writelines('\t\t<attr name="classid" kind="graph">\n')
        f.writelines('\t\t\t<string>%s</string>\n' % self.classID)
        f.writelines('\t\t</attr>\n')

        for p in range(0, newPoints.shape[0]):
            f.writelines('\t\t<node id="v_%d">\n' % p)
            f.writelines('\t\t\t<attr name="weight">\n')
            f.writelines('\t\t\t\t<string>{}</string>\n' \
                .format(
                DelaunayGenerator._weight_vector_string(newPoints[p, :])))
            f.writelines('\t\t\t</attr>\n')
            f.writelines('\t\t</node>\n')

        for p1 in range(0, newPoints.shape[0]):
            for p2 in range(p1 + 1, newPoints.shape[0]):
                if newAdjMat[p1, p2] == 1:
                    f.writelines(
                        '\t\t<edge from="v_%d" to="v_%d" isdirected="false"></edge>\n' % (
                            p1, p2))

        f.writelines('\t</graph>\n</gxl>\n')
        f.close()

        return cdg.graph.graph.Graph(filename=nameGraph + '.gxl',
                                     directory=self.path)

    @classmethod
    def generate_delaunay_adjacency(cls, points):
        noVertices = points.shape[0]

        tri = scipy.spatial.Delaunay(points)
        adjMatrix = np.zeros((noVertices, noVertices))

        # create adjacency matrix
        for t in tri.simplices:
            for i in range(0, 3):
                j = np.mod(i + 1, 3)
                adjMatrix[t[i], t[j]] = 1
                adjMatrix[t[j], t[i]] = 1

        return adjMatrix  # , degrees

    @classmethod
    def _weight_vector_string(cls, point):
        string = "["
        for e in point:
            string += str(e) + ","

        string = string[:-1] + "]"
        return string


def debug():
    gmt = cdg.graph.dissimilarity.GMT(
        executable='./graph-matching-toolkit/graph-matching-toolkit.jar')
    dataset = Delaunay('../../../DZ_Graph_Datasets/Delaunay/', dissimilarity_instance=gmt)
    dataset.load_graph_name_map()
    # dataset.gmt_executable = "../../gmt110/graph-matching-toolkit.jar"
    gmt.set_dataset_info(dataset.path)
    dataset.load_dissimilarity_matrix()
    diss = dataset.get_sub_dissimilarity_matrix(dataset.elements['10'][30: 43],
                                                dataset.elements['11'][12: 104])
    print(diss)

    fro = cdg.graph.dissimilarity.FrobeniusGraphDistance()
    dataset = KaggleSeizureDog(path='../../demo/kaggle_test', patient=1, dissimilarity_instance=fro)
    dataset.load_graph_name_map()
    fro.set_dataset_info(dataset.spektral_name)
    diss2 = dataset.get_sub_dissimilarity_matrix(dataset.elements['0'][30: 45],
                                                 dataset.elements['1'][12: 130], n_jobs=1)
    print(diss2)

    print('done')


if __name__ == "__main__":
    debug()

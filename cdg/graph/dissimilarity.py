# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Deals with the computation of dissimilarities.
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
# [1] 
#   K. Riesen, S. Emmenegger and H. Bunke. 
#   A Novel Software Toolkit for Graph Edit Distance Computation. 
#   In W.G. Kropatsch et al., editors, Proc. 9th Int. Workshop on Graph Based 
#   Representations in Pattern Recognition, LNCS 7877, 142–151, 2013.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 16/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import subprocess
import cdg.util.logger
import os
from tqdm import tqdm


class Dissimilarity(cdg.util.logger.Loggable):
    # This weird argument collect extra information required to execute `run` method
    _dataset_info = None

    def __init__(self):
        self.property = {}

    def set_properties(self, props):
        for k in props:
            self.property[k] = props[k]

    def set_dataset_info(self, value):
        self._dataset_info = value

    def run(self, source, target, graph_name_and_class,
            use_precomputed_result=False, verbose=True, id=None):
        """
        Runs the computation of dissimilarities between sources and target graphs.

        :param source: source graphs. List of graph identifiers in graph_name_map.
        :param target: target graphs. List of graph identifiers in graph_name_map.
        :param graph_name_and_class: graph-name map
        :param use_precomputed_result: True/False. All dissimilarities have been already precomputed
        :param verbose: prints the gmt stdout
        :param id: an integer used for parallel computation
        :return: (no_sources, no_targets) dissimilarity matrix
            `d_{ij} = dissimilarity(source_i,target_j)`.
        """
        if use_precomputed_result:
            self.log.warning(
                "I'm reading the result file already available, I'm not recomputing it.")
            raise NotImplementedError()

        return False

    @staticmethod
    def static_run(instance, source, target, graph_name_and_class,
                   use_precomputed_result=False, verbose=True, id=None):
        """ Static wrapper of function `run`. See `run` for documentation. """
        return instance.run(source=source, target=target,
                            graph_name_and_class=graph_name_and_class,
                            use_precomputed_result=use_precomputed_result, verbose=verbose,
                            id=id)


class FrobeniusGraphDistance(Dissimilarity):
    def run(self, source, target, graph_name_and_class,
            use_precomputed_result=False, verbose=True, id=None):

        if use_precomputed_result:
            return super().run(use_precomputed_result=use_precomputed_result, verbose=verbose)

        if len(source) == 0 or len(target) == 0:
            return np.zeros((len(source), len(target)))

        # Import data with spectral
        import spektral.datasets.kaggle_seizure
        data = spektral.datasets.kaggle_seizure.load_data(self._dataset_info)
        # Unpack data after loading
        adj, nf, ef, _, _, _, _ = data

        symmetric = False
        if len(source) == len(target):
            if np.alltrue(source == target):
                symmetric = True

        diss_mat = np.zeros((len(source), len(target)))
        for si in range(len(source)):
            target_start = 0
            if symmetric:
                target_start = si + 1
            for ti in tqdm(range(target_start, len(target))):
                # cost = np.linalg.norm(adj[si]*ef[si]-adj[ti]*ef[ti])
                # cost += np.linalg.norm(np.diag(adj[si])*nf[si]-np.diag(adj[ti])*nf[ti])
                cost = np.linalg.norm(ef[source[si]] - ef[target[ti]])
                cost += np.linalg.norm(nf[source[si]] - nf[target[ti]])
                diss_mat[si, ti] = cost
                if symmetric:
                    diss_mat[ti, si] = cost

        return diss_mat


class GMT(Dissimilarity):
    """
    The GMT class is a wrapper of the GraphMatchingToolkit (GMT).
    The standard routine can be:
        # Setup the GMT Instance
        gmt = GMT('./graph-matching-toolkit/graph-matching-toolkit.jar')
        gmt.set_database_path('../../_Graph_Database_perturbed/Letter/LOW/')
        gmt.set_input_graphs('prototypes.xml', n_p, 'target.xml', n_t)
        gmt.set_result_file('result.txt')
        # Launch GMT Procedure
        dissimilarity_matrix = gmt.launch()
    """
    # filename of the GMT executable
    gmt_executable = "./graph-matching-toolkit/graph-matching-toolkit.jar"

    def __init__(self, executable, path=None, jar=True):
        """
        Defines which is the command to launch the GMT.
        """
        super().__init__()

        self.gmt_executable = executable
        if not os.path.exists(self.gmt_executable):
            self.log.warn(self.gmt_executable + ' does not exist')

        if jar:
            self.command = "java -jar " + self.gmt_executable + " "
        else:
            self.command = "java -cp " + self.gmt_executable + " algorithm.GraphMatching "

        self.property_file = './dz_graph_GMT_propertyfile_default.prop'

        self.property['matching'] = 'VJ'
        self.property['s'] = ''
        self.property['adj'] = 'best'

        self.property['node'] = 1.0
        self.property['edge'] = 1.0

        self.property['result'] = './dz_graph_GMT_resultfile_default.txt'
        self.property['path'] = path

    def _create_property_file(self):
        f = open(self.property_file, 'w')
        for key in self.property.keys():
            f.writelines(key + " = " + str(self.property[key]) + '\n')
        f.close()

    @classmethod
    def getCxlOpening(cls):
        return '<?xml version="1.0"?>\n<GraphCollection>\n<graphs>\n'

    @classmethod
    def getCxlClosing(cls):
        return '</graphs>\n</GraphCollection>\n'

    @classmethod
    def getCxlGraphElement(cls, graphName, graphClass):
        return '<print file="' + graphName + '" class="' + graphClass + '"/>\n'

    def run(self, source, target, graph_name_and_class,
            use_precomputed_result=False, verbose=True, id=None):
        """
        Runs the GMT with the `parameters` specified, and arrange the results in a file.
        The file is then loaded and dissimilarity matrix is assembled whose entries
        are `d_{ij} = dissimilarity(source_i,target_j)`.

        :param source: source graphs. List of graph identifiers in graph_name_map.
        :param target: target graphs. List of graph identifiers in graph_name_map.
        :param use_precomputed_result: True/False. All dissimilarities have been already precomputed
        :param single_pair: if provided, it doesn't use the xml files but passes
            the single source and single target directly as a parameter.
        :param verbose: prints the gmt stdout
        :return: (no_sources, no_targets) dissimilarity matrix
        """
        if use_precomputed_result:
            return super().run(use_precomputed_result=use_precomputed_result, verbose=verbose)

        if len(source) == 0 or len(target) == 0:
            return np.zeros((len(source), len(target)))

        # setup the arguments for the graph matching toolkit
        # lists of graphs
        row_xml = id + "row.gmt"
        column_xml = id + "column.gmt"

        # Generate XML Datasets
        f_r = open(row_xml, 'w')
        f_c = open(column_xml, 'w')
        f_r.write(self.getCxlOpening())
        f_c.write(self.getCxlOpening())
        numOfRows = 0
        numOfColumns = 0
        t = 0

        for graphName in graph_name_and_class:
            f_list = []
            if t in source:
                f_list.append(f_r)
                numOfRows += 1
            if t in target:
                f_list.append(f_c)
                numOfColumns += 1

            for f in f_list:
                f.write(self.getCxlGraphElement(graphName[1], graphName[2]))

            t += 1

        f_r.write(self.getCxlClosing())
        f_c.write(self.getCxlClosing())
        f_r.close()
        f_c.close()

        self.property['source'] = row_xml
        # self.n_source = n_source
        self.property['target'] = column_xml
        # self.n_target = n_target


        # run the toolkit
        result = id + "result.gmt"
        if self._dataset_info is None:
            raise ValueError('you didn\'t provide the extra dataset info. '
                             'Use dataset.set_dataset_info(...).')
        if self._dataset_info.endswith('/'):
            self.property['path'] = self._dataset_info
        else:
            self.property['path'] = self._dataset_info + '/'
        self.property['result'] = result
        self.property_file = id + "gmt_parameters.gmt"
        self._create_property_file()
        command = self.command + self.property_file
        self.log.info("executing: " + command)
        if verbose:
            exit = subprocess.Popen(command.split()).wait()
        else:
            exit = subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()

        if exit != 0:
            raise cdg.util.errors.CDGError("Exit status={} for command: {} ".format(exit, command))

        # reimport results
        f = open(self.property['result'], 'r')
        while True:
            line = f.readline()
            if line[0:1] != '#':
                break
        dist = np.zeros((len(source), len(target)))
        r = 0
        c = 0
        while True:
            try:
                dist[r][c] = float(line)
            except ValueError:
                print(line)
                raise ValueError(line)
            c += 1
            if c >= len(target):
                c = 0
                r += 1
            if r >= len(source):
                break
            line = f.readline()

        # Clean useless file
        os.remove(self.property['source'])
        os.remove(self.property['target'])
        os.remove(self.property['result'])
        os.remove(self.property_file)

        return dist

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# This is intended as simple python interface for handling graphs in various
# formats, e.g., .gxl and .dot.
#
#
# References:
# ---------
#    [1]  Riesen, K. i Bunke, H. (2010). Graph classification and clustering
#         based on vector space embedding. World Scientific Publishing Co., Inc.
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

import xml.etree.ElementTree as et
import pydotplus
import numpy as np
import random
import os
import subprocess
import sys
import scipy.spatial as scp
import matplotlib.pyplot as plt


class Graph:
    """
    The Graph class is an interface of third-party and custom utilities. 
    Each Graph instance is intended to rely on a .gxl file.
    """

    # auxiliar
    tmp_dot_file = "tmp.dot"
    tmp_png_file = "tmp.png"
    tmp_ps_file = "tmp.ps"


    def __init__(self, filename, directory=''):
        """
        Stores the directory to the original .gxl file and the filename.
        Except for reading the `id' of the .gxl graph, the constructor only records the file path.
        """
        if directory == '':
            i = filename.rfind('/')
            directory = filename[:i]
            filename = filename[i+1:]

        if directory[len(directory)-1] != '/':
            directory += '/'


        self.directory = directory
        self.filename = filename

        self.gxlfile = self.directory + self.filename

        root = et.parse(self.gxlfile).getroot()
        self.id = root.find(".//graph").attrib["id"]
        self.adj_loaded=False


    def __str__(self):
        return self.gxlfile 


    def load(self,directed=False):

        self.load_nodes()
        self.load_adjacency(directed=directed)



    def load_nodes(self):
        tree = et.parse(self.gxlfile)

        xmlnodes = tree.findall(".//node")
        self.nodeid = []
        for n in xmlnodes:
            self.nodeid.append(n.attrib['id'])
        self.n = len(self.nodeid)


    def load_adjacency(self,directed):
        tree = et.parse(self.gxlfile)

        xmledges = tree.findall(".//edge")
        self.adj = np.zeros((self.n, self.n))  # np.array(nnodes,nnodes)
        for e in xmledges:
            fr = -1
            to = -1
            for i in range(0, self.n):
                if e.attrib['from'] == self.nodeid[i]:
                    fr = i
                if e.attrib['to'] == self.nodeid[i]:
                    to = i
            self.adj[fr][to] = 1
            if ~directed:
                self.adj[to][fr] = 1

        self.adj_loaded=True



    def pydot(self):
        """
        Returns a instance of  pydotplus.graphviz graph.
        """
        command = "gxl2dot -o "+ Graph.tmp_dot_file + " " + self.directory + self.filename
        print("executing: " + command)
        # subprocess.call(command.split())
        subprocess.Popen(command.split()).wait()

        self.pydotgraph = pydotplus.graphviz.graph_from_dot_file(Graph.tmp_dot_file)
        os.remove(Graph.tmp_dot_file)

        return self.pydotgraph


    def get_density(self):
        self.load()
        density = 1.
        density *= sum(sum(self.adj))
        if self.adj.shape[0]<2:
            return 0
        density /= self.adj.shape[0]*(self.adj.shape[0]-1)
        return density


    def get_laplacian_spec_gap(self):
        self.load()
        if self.adj.shape[0]<1:
            return 0
        laplacian = 0. - self.adj
        for i in range(0,self.adj.shape[0]):
            laplacian[i][i]=sum(self.adj[i][:])

        eig, _ = np.linalg.eig(laplacian)

        if self.adj.shape[0]<2:
            return np.absolute(eig[0])
        return np.absolute(eig[0])-np.absolute(eig[1])






class Prototype:


    @classmethod
    def median(cls,dissimilarity_matrix):
        """
        Median graph. See Definition 6.4 in [1].
        """
        median = -1
        ct = -1
        median_sum = np.inf
        for row in dissimilarity_matrix:
            ct += 1
            row_sum = sum(row)
            if row_sum < median_sum:
                median = ct
                median_sum = row_sum

        return median

    @classmethod
    def center(cls,dissimilarity_matrix):
        """
        Center graph. See Definition 6.4 in [1].
        """

        center = -1
        ct = -1
        center_max = np.inf
        for row in dissimilarity_matrix:
            ct += 1
            row_max = max(row)
            if row_max < center_max:
                center = ct
                center_max = row_max

        return center

    @classmethod
    def marginal(cls,dissimilarity_matrix):
        """
        Marginal graph. See Definition 6.4 in [1].
        """
        marg = -1
        ct = -1
        marg_sum = 0
        for row in dissimilarity_matrix:
            ct += 1
            row_sum = sum(row)
            if row_sum > marg_sum:
                marg = ct
                marg_sum = row_sum

        return marg



    @classmethod
    def spanning(cls,dissimilarity_matrix, n_prototypes=3):
        """
        The first prototype is the set median graph, each additional prototype
        selected by the spanning prototype selector is the graph furthest away
        from the already selected prototype graphs [1].

        :param dissimilarity_matrix:
        :param n_prototypes:
        :return:
        """

        nr, nc = dissimilarity_matrix.shape
        prototypes = []
        prototypes.append(Prototype.median(dissimilarity_matrix))

        val_hat = 0

        for pi in range(1, n_prototypes):

            val = 0

            for i in range(1, nc):

                # controlla se gia selezionato
                found = False
                for p in prototypes:
                    if p == i:
                        found = True
                        break

                # trova il migliore
                if not found:
                    tmp = min(dissimilarity_matrix[i, prototypes])
                    if tmp > val:
                        val = tmp
                        p_new = i

            if val>val_hat:
                val_hat = val

            # aggiorna
            prototypes.append(p_new)

        return prototypes, val_hat

    @classmethod
    def k_centers(cls,dissimilarity_matrix, n_prototypes=3):

        nr, nc = dissimilarity_matrix.shape

        # init prototypes
        datapoints = [i for i in range(0, nr)]
        prototypes = np.random.choice(datapoints, n_prototypes, False)

        itermax = 100
        ct = 0
        changed = True
        while changed and ct < itermax:

            changed = False

            # assegna classe
            c = [[] for p in prototypes]
            for d in datapoints:
                dist = np.inf
                c_tmp = 0
                for pi in range(0, n_prototypes):
                    if dissimilarity_matrix[prototypes[pi], d] < dist:
                        dist = dissimilarity_matrix[prototypes[pi], d]
                        c_tmp = pi
                c[c_tmp].append(d)

            # aggiorna prototipi
            val = [np.inf for p in prototypes]

            for pi in range(0, n_prototypes):
                p_new = prototypes[pi]
                for d in c[pi]:
                    tmp = max(dissimilarity_matrix[d, c[pi]])
                    if tmp < val[pi]:
                        val[pi] = tmp
                        p_new = d
                if prototypes[pi] != p_new:
                    prototypes[pi] = p_new
                    changed = True

            ct += 1
        return prototypes, max(val)

    @classmethod
    def MP(cls, dissimilarity_matrix, n_prototypes=3, display_values=False, value_fun="ell-1"):
        """
        Similar to the Matching Pursuit.

        prototypes = []
        while length(prototypes) < n_prototypes
            \\ell 1
            new_prototype = arg min { sum_t [ min_p d(p,t) ] }
            \\min
            new_prototype = arg min { max_t [ min_p d(p,t) ] }
            \\min max
            new_prototype = arg min { sum_t [ min_p d(p,t) ] - sum_p d(p,p')}

            prototypes.append( new_prototype )
        """
        nr, nc = dissimilarity_matrix.shape

        # init set of prototype candidates
        T = []
        for n in range(0, nr):
            T.append(n)

        # # Compute Prototype Set
        # _hat = best so far
        # _bar = current candidate
        P = []
        V = []
        # cycle until we have the num of prototypes we want
        cycles = n_prototypes
        if display_values:
            cycles = nr

        for n in range(0, cycles):
            val_hat = np.inf
            p_hat = -1

            # sweep all candidates
            for p_bar in T:
                # test on a candidate
                P.append(p_bar)

                # assess temporary maximal distance
                #   find the minima in columns of diss_mat[P][:]
                #   sum the minima
                if value_fun == "ell-1":
                    # ell-1
                    val = sum(np.min(dissimilarity_matrix[P, :], axis=0))
                    # for t in T:
                    #     val += min(dissimilarity_matrix[p][t] for p in P)
                elif value_fun == "min":
                    # max dist
                    val = np.max(np.min(dissimilarity_matrix[P, :], axis=0))
                elif value_fun == "min-max":
                    # max dist
                    val = np.max(np.min(dissimilarity_matrix[P, :], axis=0))
                    val -= sum(dissimilarity_matrix[p_bar, P])
                else:
                    raise ValueError("The value function '" + value_fun + "' not recognized")

                # save the maximum so far
                if val < val_hat:
                    val_hat = val
                    p_hat = p_bar

                # remove the tested candidate
                P.remove(p_bar)

            # status bar
            sys.stdout.write("\rMP statusbar " + str(n))
            sys.stdout.flush()

            # once a candidate has been selected, update the sets
            P.append(p_hat)
            T.remove(p_hat)
            V.append(val_hat)

        if display_values:
            plt.plot([i for i in range(0, n_prototypes)], V[:n_prototypes], 'g*', label="taken")
            plt.plot([i for i in range(n_prototypes, cycles)], V[n_prototypes:], 'r*', label="discarded")
            plt.title(value_fun)
            plt.show()

        # check for a better set of prototypes
        amv = np.argmin(V[:n_prototypes])
        n_prot_suggested = n_prototypes
        if n_prot_suggested > amv + 1:
            print("The suggested number of prototype is ", amv + 1)

        # this is necessary in case we have continued exploring
        P = P[:n_prototypes]

        return P, val_hat


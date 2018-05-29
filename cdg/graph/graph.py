# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# This is intended as simple python interface for handling graphs in various
# formats, e.g., .gxl and .dot.
#
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 20/01/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import xml.etree.ElementTree as et
import pydotplus
import numpy as np
import os
import subprocess


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
            if i >= 0:
                directory = filename[:i]
                filename = filename[i + 1:]
            else:
                directory = './'

        if directory[-1] != '/':
            directory += '/'

        self.directory = directory
        self.filename = filename

        self.gxlfile = self.directory + self.filename

        root = et.parse(self.gxlfile).getroot()
        self.id = root.find(".//graph").attrib["id"]
        self.adj_loaded = False

    def __str__(self):
        return self.gxlfile

    def load(self, directed=False):

        self.load_nodes()
        self.load_adjacency(directed=directed)

    def load_nodes(self):
        tree = et.parse(self.gxlfile)

        xmlnodes = tree.findall(".//node")
        self.nodeid = []
        for n in xmlnodes:
            self.nodeid.append(n.attrib['id'])
        self.n = len(self.nodeid)

    def load_adjacency(self, directed):
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

        self.adj_loaded = True

    def pydot(self):
        """
        Returns a instance of  pydotplus.graphviz graph.
        """
        command = "gxl2dot -o " + Graph.tmp_dot_file + " " + self.directory + self.filename
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
        if self.adj.shape[0] < 2:
            return 0
        density /= self.adj.shape[0] * (self.adj.shape[0] - 1)
        return density

    def get_laplacian_spec_gap(self):
        self.load()
        if self.adj.shape[0] < 1:
            return 0
        laplacian = 0. - self.adj
        for i in range(0, self.adj.shape[0]):
            laplacian[i][i] = sum(self.adj[i][:])

        eig, _ = np.linalg.eig(laplacian)

        if self.adj.shape[0] < 2:
            return np.absolute(eig[0])
        return np.absolute(eig[0]) - np.absolute(eig[1])

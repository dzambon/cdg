# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Generate Delaunay dataset [ssci17].
#
#
# References:
# ---------
# [ssci17] 
#   Detecting Changes in Sequences of Attributed Graphs.
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   IEEE Symposium Series on Computational Intelligence, 2017.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 25/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cdg.graph.database

# Parameters
no_nodes = 10
no_graphs_per_class = 100
classes = [0, 2, 4, 6, 8]
path = "./delaunay"

# Create dataset
dataset = cdg.graph.database.Delaunay(path)
dataset.generate_new_dataset(seed_points=no_nodes, classes=classes, no_graphs=no_graphs_per_class, format=['gxl', 'npy'])
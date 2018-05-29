# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Tests in [ijcnn17]. This file has to be placed in `cdg/simulation/` folder.
#
# References:
# ---------
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
#
# -----------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# ------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 25/04/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cdg.simulation.simulations


class SimulationDissRep_vec(cdg.simulation.simulations.SimulationPrototypeBased,
                            cdg.simulation.simulations.SimulationCLT):
    pass


class SimulationFeature_scalar(cdg.simulation.simulations.SimulationFeature,
                               cdg.simulation.simulations.SimulationTwoSided):
    pass

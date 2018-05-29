# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Compute the gaussian thresholds once for all.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 27/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from cdg.changedetection.cusum import GaussianCusum as gcusum
import joblib 

dofs = [1, 2, 3, 4, 8, 10, 15]
res = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(gcusum.precomp_threshold)(dof=[dofs[i]], len_sim=5e4) for i in range(len(dofs)))


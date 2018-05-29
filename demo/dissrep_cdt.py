# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# CUSUM-like change detection test.
#
#
# References:
# ---------
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 18/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cdg.changedetection.cusum
import cdg.embedding.embedding
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import scipy.spatial.distance

# create two multivariate distributions
sample_size = 5000
rv1 = multivariate_normal(mean=[0., 0.], cov=[[1., 0.], [0., .2]])
rv2 = multivariate_normal(mean=[0., 0.], cov=[[1., 0.], [0., 2.]])
training_stream = rv1.rvs(size=1000)
x1 = rv1.rvs(size=int(sample_size / 5 * 4))
x2 = rv2.rvs(size=int(sample_size / 5))
test_stream = np.concatenate((x1, x2), axis=0)

# create dissimilarity matrix
diss_mat = scipy.spatial.distance.cdist(training_stream, training_stream, metric='euclidean')
# init dissimilarity representation
diss_rep = cdg.embedding.embedding.DissimilarityRepresentation()
diss_rep.set_parameters(M=3)
diss_rep.fit(dissimilarity_matrix=diss_mat)
# create representations
y = scipy.spatial.distance.cdist(test_stream, training_stream[diss_rep.prototype_indices], metric='euclidean')

# change detection test applied to window data
cdt = cdg.changedetection.cusum.GaussianCusum(arl=100, window_size=10)
cdt.fit(y[:100], estimate_threshold=True, len_simulation=1000)
labels, cumulative_sums = cdt.predict(y[100:], reset=False)

# plot
plt.plot(labels * max(cumulative_sums), '+k', label='prediction')
plt.plot(cumulative_sums, label='culative sum')
plt.plot([cdt.threshold] * len(cumulative_sums), label='threshold')
plt.grid(True)
plt.legend()
plt.show()

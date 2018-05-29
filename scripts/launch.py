# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Run some experiments in cdg.simulation.simulations.
# The datasets already available are Letter, Mutaganicity and AIDS by IAM [1], 
# and Delaunay [ssci17] and Markov [2].
#
#
# References:
# ---------
#
# [tnnls17]
#   Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.
#   Concept Drift and Anomaly Detection in Graph Streams.
#   IEEE Transactions on Neural Networks and Learning Systems (2018).
#
# [ssci17] 
#   Detecting Changes in Sequences of Attributed Graphs.
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   IEEE Symposium Series on Computational Intelligence (2017).
#
# [ijcnn18]
#   Zambon, Daniele, Lorenzo Livi, and Cesare Alippi.
#   Anomaly and Change Detection in Graph Streams through Constant-Curvature
#   Manifold Embeddings.
#   IEEE International Joint Conference on Neural Networks (2018).
#
# [1]
#   Riesen, Kaspar, and Horst Bunke.
#   IAM graph database repository for graph based pattern recognition and
#   machine learning.
#   Structural, Syntactic, and Statistical Pattern Recognition (2008).
#
# [2]
#   Livi, Lorenzo, Antonello Rizzi, and Alireza Sadeghian.
#   Optimized dissimilarity space embedding for labeled graphs.
#   Information Sciences (2014).
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
last_update = '27/05/2018'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import sys
import argparse
import cdg
import cdg.util.logger
import cdg.embedding.embedding
import cdg.embedding.manifold
import cdg.embedding.feature
import cdg.graph.database
import cdg.simulation.parameters
import datetime

###############################################################################
# Edit this part
###############################################################################

# Set the debuggin options
cdg.util.logger.set_stdout_level(cdg.util.logger.DEBUG)
log_file = cdg.util.logger.set_filelog_level(
    level=cdg.util.logger.DEBUG,
    name=datetime.datetime.now().strftime('%G%m%d%H%M-%f'))
cdg.util.logger.enable_logrun(level=False)


# Defines your custom parameter settings if you want
def setting_custom(pars):
    new_pars = setting_toy(pars)
    return new_pars


# Defines the datasets location
cdg_path = os.path.dirname(os.path.realpath(__file__)) + '/..'
_fro = cdg.graph.dissimilarity.FrobeniusGraphDistance()
_gmt = cdg.graph.dissimilarity.GMT(executable=cdg_path+"/graph-matching-toolkit/graph-matching-toolkit.jar")


def init_delaunay():
    path = "./datasets/delaunay"
    dataset = cdg.graph.database.Delaunay(path=path, dissimilarity_instance=_gmt)
    _gmt.set_dataset_info(dataset.path)
    return dataset

def init_markov(difficulty):
    if difficulty is None:
        difficulty=2
    path = "./datasets/markov"
    dataset = cdg.graph.database.Markov(path=path, difficulty=difficulty, dissimilarity_instance=_gmt)
    _gmt.set_dataset_info(dataset.path)
    return dataset

def init_letter(distortion):
    if distortion is None:
        distortion="HIGH"
    path = "./datasets/iam/Letter"
    dataset = cdg.graph.database.Letter(path=path, distortion=distortion, dissimilarity_instance=_gmt)
    _gmt.set_dataset_info(dataset.path)
    return dataset


def init_mutagenicity():
    path = "./datasets/iam/Mutagenicity/data"
    dataset = cdg.graph.database.Mutagenicity(path=path, dissimilarity_instance=_gmt)
    _gmt.set_dataset_info(dataset.path)
    return dataset


def init_aids():
    path = "./datasets/iam/AIDS/data"
    dataset = cdg.graph.database.AIDS(path, dissimilarity_instance=_gmt)
    _gmt.set_dataset_info(dataset.path)
    return dataset


###############################################################################
# You are not supposed to edit this part
###############################################################################

# Parameter settings
# ------------------

def setting_toy(pars):
    cdg.util.logger.glog().warning('**************** toy settings *********')
    pars.significance_level = 0.01

    pars.train_embedding_t = 100
    pars.train_changedetection_t = 300

    pars.test_nominal_t = int(pars.arl_t() * 2)
    pars.test_drift_t = int(pars.arl_t() * 0)
    pars.test_nonnominal_t = int(pars.arl_t() * 2)

    pars.no_simulations_thresh_est = 1000
    pars.beta = 0.6

    return pars


def setting_IJCNN18(pars):
    """ Settings declared in [ijcnn18]. """
    pars.significance_level = 0.01

    pars.train_embedding_t = 300
    pars.train_changedetection_t = int(pars.arl_t() * 6)

    pars.test_nominal_t = int(pars.arl_t() * 6)
    pars.test_drift_t = int(pars.arl_t() * 0)
    pars.test_nonnominal_t = int(pars.arl_t() * 6)

    return pars


def setting_SSCI17(pars):
    """
    Settings declared in [ssci17].

    Differently from what stated in the paper, the results have been erroneously generated
    by sequences shorter than reported.
    However, the outcome in the declared setting is qualitatively the same. I point out that:
        - The training sequence was shorter, hence the reported outcome may be seen as a
        lower bound on the actual performance.
        - The numerical estimation of threshold is less accurate.
        - The short testing sequence yields not very accurate estimates of the ARL0 and ARL1.
        - The comparison in there is consistent since all experiments have the same settings,
        and despite the above points, the outcome in the declared settings appears to be
        qualitatively equivalent to the reported one.
    The actual settings are reported in function `setting_SSCI17_2`.

    Apologies of the inconvenience.
    """
    pars.significance_level = 0.01

    pars.train_embedding_t = 300
    pars.train_changedetection_t = 1000

    pars.test_nominal_t = int(pars.arl_t() * 8)
    pars.test_drift_t = int(pars.arl_t() * 0)
    pars.test_nonnominal_t = int(pars.arl_t() * 4)

    return pars


def setting_SSCI17_2(pars):
    """ Settings for the results reported in [ssci17]. """
    pars.significance_level = 0.01

    pars.train_embedding_t = 100
    pars.train_changedetection_t = 300

    pars.test_nominal_t = int(pars.arl_t() * 4)
    pars.test_drift_t = int(pars.arl_t() * 0)
    pars.test_nonnominal_t = int(pars.arl_t() * 2)

    pars.no_simulations_thresh_est = 10000

    return pars


def setting_TNNLS17(pars):
    """ Settings declared in [tnnls17]. """
    pars.significance_level = 0.005

    pars.train_embedding_t = 300
    pars.train_changedetection_t = 1000

    pars.test_nominal_t = int(pars.arl_t() * 20)
    pars.test_drift_t = int(pars.arl_t() * 0)
    pars.test_nonnominal_t = int(pars.arl_t() * 12)

    return pars


# Available options for the arguments 
# -----------------------------------

# Settings
SETTING_TOY = ('toy', setting_toy)
SETTING_TNNLS17 = ('tnnls17', setting_TNNLS17)
SETTING_SSCI17 = ('ssci17', setting_SSCI17)
SETTING_SSCI17_2 = ('ssci17_2', setting_SSCI17_2)
SETTING_IJCNN18 = ('ijcnn18', setting_IJCNN18)
SETTING_CUSTOM = ('custom', setting_custom)
_settings = [SETTING_TOY, SETTING_TNNLS17, SETTING_SSCI17, SETTING_SSCI17_2, SETTING_IJCNN18,
             SETTING_CUSTOM]
available_settings = {}
for _set in _settings:
    available_settings[_set[0]] = _set[1]

# Datasets
LETTER = 'Letter'
MUTAGENICITY = 'Mutagenicity'
AIDS = 'AIDS'
DELAUNAY = 'Delaunay'
MARKOV = 'Markov'
available_datasets = [LETTER, MUTAGENICITY, AIDS, DELAUNAY, MARKOV]

# Embedding techniques
DISSREP = ('DissRep', cdg.embedding.embedding.DissimilarityRepresentation)
EUCLIDEAN_DM = ('EuclideanDM', cdg.embedding.manifold.EuclideanDR)
SPHERICAL_DM = ('SphericalDM', cdg.embedding.manifold.SphericalDR)
HYPERBOLIC_DM = ('HyperbolicDM', cdg.embedding.manifold.HyperbolicDR)
EUCLIDEAN = ('Euclidean', cdg.embedding.manifold.EuclideanDR)
SPHERICAL = ('Spherical', cdg.embedding.manifold.SphericalDR)
HYPERBOLIC = ('Hyperbolic', cdg.embedding.manifold.HyperbolicDR)
GRAPHMU = ('GraphMu', cdg.embedding.feature.DistanceGraphMean)
DENSITY = ('Density', cdg.embedding.feature.Density)
SPECGAP = ('SpecGap', cdg.embedding.feature.SpectralGap)
_manifolds = [DISSREP, \
              EUCLIDEAN_DM, SPHERICAL_DM, HYPERBOLIC_DM, \
              EUCLIDEAN, SPHERICAL, HYPERBOLIC, \
              GRAPHMU, DENSITY, DENSITY, SPECGAP]
available_manifolds = {}
for _man in _manifolds:
    available_manifolds[_man[0]] = _man[1]

# Define the argument parser
parser = argparse.ArgumentParser(description='')
# parser.add_argument('-e', '--experiment', type=str, help='experiment name')
parser.add_argument('-e', '--experiment', type=str,
                    required=True,
                    choices=available_datasets,
                    help='experiment name')
parser.add_argument('-p', '--patient', type=str,
                    help='patient id, or subdataset id')
parser.add_argument('-m', '--manifold', type=str,
                    default=DISSREP[0],
                    choices=available_manifolds.keys(),
                    help='type of the manifold')
parser.add_argument('-d', '--manifoldDimension', type=int,
                    help='dimension of the manifold')
parser.add_argument('-M', '--noPrototypes', type=int, help='number of prototypes')
parser.add_argument('-n', '--winSize', type=int, default=1, help='size of the testing window')
parser.add_argument('-N', '--class0str', type=str, help='nominal class')
parser.add_argument('-A', '--class1str', type=str, help='non-nominal class')
parser.add_argument('-r', '--numSimulations', type=int, default=15,
                    help='number of repreated simulations')
parser.add_argument('-s', '--seed', type=int, help='seed for the PRG')
parser.add_argument('-f', '--folderPrefix', type=str, default='cdgexp',
                    help='prefix for the folder with the results')
parser.add_argument('-S', '--settings', type=str, default='toy',
                    choices=available_settings,
                    help='other settings for the experiment')


# Main function 
# -------------


def init_dataset(experiment, class0str, class1str, patient=None):
    if experiment == AIDS:
        class0 = ['i']
        class1 = ['a']
        if class0str is not None:
            class0 = [class0str]
            if class0 == ['a']:
                class1 = ['i']
            else:
                class1 = ['a']
        dataset = init_aids()

    elif experiment == MUTAGENICITY:
        class0 = ['nonmutagen']
        class1 = ['mutagen']
        if class0str is not None:
            class0 = [class0str]
            if class0 == ['mutagen']:
                class1 = ['nonmutagen']
            else:
                class1 = ['mutagen']
        dataset = init_mutagenicity()

    elif experiment == LETTER:
        class0 = ['A', 'E', 'F', 'H']
        class1 = ['F', 'H', 'I', 'K']
        if class0str is not None:
            class0 = [c for c in class0str]
        if class0str is not None:
            class1 = [c for c in class1str]
        dataset = init_letter(patient)

    elif experiment == DELAUNAY:
        class0 = ['0']
        class1 = ['6']
        if class0str is not None:
            class0 = [class0str]
        if class1str is not None:
            class1 = [class1str]
        dataset = init_delaunay()

    elif experiment == MARKOV:
        class0 = ['0']
        class1 = ['1']
        if class0str is not None:
            class0 = [class0str]
        if class1str is not None:
            class1 = [class1str]
        dataset = init_markov(patient)

    else:
        raise ValueError('dataset {} not available'.format(experiment))

    dataset.load_dissimilarity_matrix()

    return dataset, class0, class1


opening_text = '***************************\n'
opening_text += '* CDG Experiments         *\n'
opening_text += '***************************\n'
opening_text += 'Datasets available: %s.\n' % str(available_datasets)
opening_text += 'Author: Daniele Zambon\n'
opening_text += 'eMail: daniele.zambon@usi.ch\n'
opening_text += 'last update: %s\n\n' % last_update


def main(argv):
    print(opening_text)
    args = parser.parse_args()

    # command to parse
    launchingCommand = ''
    for a in argv:
        launchingCommand += ' ' + a
    import cdg
    import cdg.util.logger
    #print(cdg.util.logger._stdout_level)
    cdg.util.logger.glog().info("command: " + launchingCommand)

    # Default parameters
    pars = cdg.simulation.parameters.ParametersOnline()
    pars.cdg_commit_hash = cdg.version()[0]

    # check problems
    if args.noPrototypes is not None:
        if args.noPrototypes <= 0:
            raise ValueError("no prototypes must be positive")
    if args.manifoldDimension is not None:
        if args.manifoldDimension <= 0:
            raise ValueError("manifoldDimension must be positive")

    # User-defined parameters
    dataset, class0, class1 = init_dataset(args.experiment, args.class0str, args.class1str, args.patient)
    pars.class0 = class0
    pars.class1 = class1

    pars.name_nominal = ''
    for c in pars.class0:
        pars.name_nominal += c

    pars.name_nonnominal = ''
    for c in pars.class1:
        pars.name_nonnominal += c

    pars.launching_command = launchingCommand

    man_tmp = available_manifolds[args.manifold]()

    if args.manifold == DISSREP[0]:
        import cdg.simulation.simulations_tnnls17
        if args.noPrototypes is None:
            raise ValueError('You didn\'t provide a number of prototypes for the dissimilarity '
                             'representation.')
        if args.manifoldDimension is not None \
                and args.noPrototypes != args.manifoldDimension:
            raise NotImplementedError('Number of prototypes {} different from man. dimension {}. '
                                      'This is not allowed in dissimilarity representation.'
                                      .format(args.noPrototypes, args.manifoldDimension))
        man_tmp.set_parameters(M=args.noPrototypes)
        simulation = cdg.simulation.simulations_tnnls17.SimulationDissRep_vec()
        pars.window_size = args.winSize

    elif args.manifold == GRAPHMU[0]:
        import cdg.simulation.simulations_ijcnn18
        man_tmp.set_parameters(M=args.manifoldDimension)
        simulation = cdg.simulation.simulations_ijcnn18.SimulationGraphSpace_scalar()
        if args.winSize != 1:
            print('forcing window size to 1')
        pars.window_size = 1

    elif args.manifold in [EUCLIDEAN_DM[0], SPHERICAL_DM[0], HYPERBOLIC_DM[0]]:
        if args.noPrototypes is None:
            args.noPrototypes = args.manifoldDimension + 1
        import cdg.simulation.simulations_ijcnn18
        man_tmp.set_parameters(d=args.manifoldDimension, M=args.noPrototypes)
        simulation = cdg.simulation.simulations_ijcnn18.SimulationManifold_scalar()
        if args.winSize != 1:
            print('forcing window size to 1')
        pars.window_size = 1

    elif args.manifold in [DENSITY[0], SPECGAP[0]]:
        import cdg.simulation.simulations_tnnls17
        man_tmp.set_parameters()
        simulation = cdg.simulation.simulations_tnnls17.SimulationFeature_scalar()
        if args.winSize != 1:
            print('forcing window size to 1')
        pars.window_size = 1

    else:
        raise ValueError('manifold {} nor recognized or not implemented yet.'.format(args.manifold))

    pars.manifold = man_tmp

    # Forced parameters
    pars = available_settings[args.settings](pars)

    # Make the parameters not modifiable
    pars.freeze()

    # Output folder
    folder = args.folderPrefix + \
             '_' + pars.creation_time.strftime('%G%m%d_%H%M_%S') + \
             '_' + dataset.name + \
             '_' + str(pars.manifold) + \
             '_n' + str(pars.window_size) + \
             '_' + pars.name_nominal + \
             '_' + pars.name_nonnominal

    # Run simulation
    simulation.set(pars, dataset, args.numSimulations, folder)
    simulation.run(seed=args.seed, logfile=log_file)


if __name__ == "__main__":
    main(sys.argv[1:])

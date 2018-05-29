# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Setup the datasets.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
last_update = '24/05/2018'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import sys
import argparse
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

# Set the debugging options
cdg.util.logger.set_stdout_level(cdg.util.logger.INFO)
log_file = cdg.util.logger.set_filelog_level(
    level=cdg.util.logger.DEBUG,
    name=datetime.datetime.now().strftime('%G%m%d%H%M-%f'))
cdg.util.logger.enable_logrun(level=False)


# Defines the datasets location
def get_elements_delaunay(elements):
    return elements['0'], elements['0'] + elements['2'] + elements['4'] + elements['6'] \
                                        + elements['8'] + elements['10'] + elements['12']

def get_elements_delaunay_8_12(elements):
    return elements['0'], elements['0'] + elements['8'] + elements['9'] + elements['10'] \
                                        + elements['11'] + elements['12']

def get_elements_markov(elements):
    return elements['0'], elements['0'] + elements['1']

def get_elements_letter(elements):
    return elements['A'] + elements['E'] + elements['F'] + elements['H'] + elements['I'], \
           elements['A'] + elements['E'] + elements['F'] + elements['H'] + elements['I'] \
           + elements['K'] + elements['L'] + elements['M'] + elements['N'] + elements['T']

def get_elements_mutag(elements):
    return elements['nonmutagen'], elements['nonmutagen'] + elements['mutagen']

def get_elements_aids(elements):
    return elements['i'], elements['i'] + elements['a']

def get_elements_dog(elements):
    return elements['0'], elements['0'] + elements['1']


# gmt_executable = "./graph-matching-toolkit/graph-matching-toolkit.jar"

###############################################################################
# You are not supposed to edit this part
###############################################################################

# Available options for the arguments
# -----------------------------------

# Datasets
_use_gmt = True
_class_index = 0
_get_elements_index = 1
_use_gmt_index = 2
LETTER = ('Letter', cdg.graph.database.Letter, get_elements_letter, _use_gmt)
MUTAGENICITY = ('Mutagenicity', cdg.graph.database.Mutagenicity, get_elements_mutag, _use_gmt)
AIDS = ('AIDS', cdg.graph.database.AIDS, get_elements_aids, _use_gmt)
DELAUNAY = ('Delaunay', cdg.graph.database.Delaunay, get_elements_delaunay, _use_gmt)
DELAUNAY812 = ('Delaunay812', cdg.graph.database.Delaunay, get_elements_delaunay_8_12, _use_gmt)
MARKOV = ('Markov', cdg.graph.database.Markov, get_elements_markov, _use_gmt)
KAGGLEDOG = ('Kaggle', cdg.graph.database.KaggleSeizureDog, get_elements_dog, not _use_gmt)
_datasets = [LETTER, MUTAGENICITY, AIDS, DELAUNAY, DELAUNAY812, MARKOV, KAGGLEDOG]
available_datasets = {}
for _set in _datasets:
    available_datasets[_set[0]] = _set[1:]

# Define the argument parser
parser = argparse.ArgumentParser(description='')
# parser.add_argument('-e', '--experiment', type=str, help='experiment name')
parser.add_argument('-d', '--dataset', type=str,
                    required=True,
                    choices=available_datasets.keys(),
                    help='dataset name')
parser.add_argument('-f', '--folder', type=str, default='./tmp',
                    help='folder of the dataset')
parser.add_argument('-s', '--subdataset', type=str, default=None,
                    help='sometimes is needed')
parser.add_argument('-j', '--noJobs', type=int, default=-1,
                    help='number of jobs of joblib')

cdg_path = os.path.dirname(os.path.realpath(__file__)) + '/..'
parser.add_argument('--gmtpath', type=str,
                    default=cdg_path+"/graph-matching-toolkit/graph-matching-toolkit.jar",
                    help='path to the graph matching toolkit')

# Main function 
# -------------


opening_text = '***************************\n'
opening_text += '* CDG Prepare dataset    *\n'
opening_text += '***************************\n'
opening_text += 'Datasets available: %s.\n' % str(available_datasets.keys())
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
    cdg.util.logger.glog().info("command: " + launchingCommand)

    sel_dataset = available_datasets[args.dataset]
    if sel_dataset[_use_gmt_index]:
        diss_object = cdg.graph.dissimilarity.GMT(executable=args.gmtpath)
    else:
        diss_object = cdg.graph.dissimilarity.FrobeniusGraphDistance()

    if sel_dataset[_class_index] in [cdg.graph.database.KaggleSeizureDog]:
        dataset = sel_dataset[_class_index](path=args.folder, patient=int(args.subdataset),
                                            dissimilarity_instance=diss_object)
    elif sel_dataset[_class_index] in [cdg.graph.database.Letter]:
        dataset = sel_dataset[_class_index](path=args.folder, distortion=args.subdataset,
                                            dissimilarity_instance=diss_object)
    elif sel_dataset[_class_index] in [cdg.graph.database.Markov]:
        dataset = sel_dataset[_class_index](path=args.folder, difficulty=args.subdataset,
                                            dissimilarity_instance=diss_object, dot_to_gxl=True)
    else:
        dataset = sel_dataset[_class_index](path=args.folder,
                                            dissimilarity_instance=diss_object)
    dataset.load_graph_name_map()
    if sel_dataset[_use_gmt_index]:
        diss_object.set_dataset_info(dataset.path)
    else:
        diss_object.set_dataset_info(dataset.spektral_name)

    element0, element1 = sel_dataset[_get_elements_index](dataset.elements)
    di = dataset.get_sub_dissimilarity_matrix(element0, element1, n_jobs=args.noJobs)
    import numpy as np
    prc = sum(sum(dataset.dissimilarity_matrix >= 0)) * 1. / dataset.dissimilarity_matrix.shape[
                                                                 0] ** 2
    cdg.util.logger.glog().info('Pencentage of computed distanace {:.3f}%'.format(prc * 100))


if __name__ == "__main__":
    main(sys.argv[1:])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# This demo script launch the experiments in simulations.py.
# The datasets already considered are Letter, Mutaganicity and AIDS in [1].
#
#
# References:
# ---------
# [1] Riesen, Kaspar, and Horst Bunke. "IAM graph database repository for graph
#     based pattern recognition and machine learning." Structural, Syntactic,
#     and Statistical Pattern Recognition (2008): 287-297.
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


import sys
import getopt
import matplotlib
matplotlib.use('Agg')
from cdg import database
from cdg import simulations


def init_letter():

    distortion = "HIGH"
    path = "../../IAM_Graph_Database/Letter/Letter"
    dataset = database.Letter(path,distortion)

    return dataset


def init_mutagenicity():

    path = "../../IAM_Graph_Database/Mutagenicity/Mutagenicity/data"
    dataset = database.Mutagenicity(path)

    return dataset


def init_aids():

    path = "../../IAM_Graph_Database/AIDS/AIDS/data"
    dataset = database.AIDS(path)

    return dataset


def init_del():

    path = "../../DZ_Graph_Datasets/Delaunay"
    dataset = database.Delaunay(path)

    return dataset



def setting_toy(pars):

    pars.arl_w = 100

    pars.trainProtSel_t = 100 
    pars.trainCusum_t   = 300 

    pars.testNominal_t    = pars.arl_t()*4
    pars.testDrift_t      = pars.arl_t()*0
    pars.testNonnominal_t = pars.arl_t()*2

    pars.noSimulationsThreshEstimation = 10000
    pars.lenSimulationThreshEstimation = pars.testTotal_w() + 1

    return pars




# # # # # # # # # # # # # # #
# # launch the simulation # #
# # # # # # # # # # # # # # #



def main(argv):
    print('**************************')
    print('* CDG -- demo            *')
    print('**************************\n')
    print('Datasets available: Letter, Mutaganicity, AIDS and Delaunay.')
    print('')
    print('Author: Daniele Zambon')
    print('eMail: daniele.zambon@usi.ch\n')
    print('last update: 23/08/2017')


    helpMenu = 'Three examples on how to launch an experiment:\n'
    # helpMenu += '\t-----------------------------------\n'
    helpMenu += '\tpython3 demo.py -e AIDS\n'
    helpMenu += '\tpython3 demo.py -e Mutagenicity -M 8 -n nonmutagenic\n'
    helpMenu += '\tpython3 demo.py -e Letter -r 10 -M 3 -D 25 -n AEFH -a FHIK\n'
    # helpMenu += '\t------------------------------------\n\n'
    helpMenu += '\n'
    helpMenu += 'List of options:\n'
    helpMenu += '-h\tprint this help\n'
    helpMenu += '-e\texperiment name (Letter|Mutagenicity|AIDS|Delaunay)\n'
    helpMenu += '-M\tnumber of prototypes\n'
    helpMenu += '-D\tsize of the testing window\n'
    helpMenu += '-n\tnominal class\n'
    helpMenu += '-a\tout-of-control class\n'
    helpMenu += '-r\tnumber of repeated simulations\n'
    helpMenu += '[-c\ttype of change (under development)]\n'



    # Default settings
    numSimulations = 15
    mainSeed = 123
    changeType = 'abrupt'
    pars = simulations.Parameters()
    gmt_executable = "../../GMT/dz_GraphMatching.jar"

    launchingCommand = ''
    for a in argv:
        launchingCommand += ' '+a
    print(launchingCommand)

    try:
        opts, args = getopt.getopt(argv,"e:n:a:M:D:r:s:c:h")
    except getopt.GetoptError:
        print(helpMenu)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpMenu)
            sys.exit()
        elif opt in ("-e"):
            experiment = arg
        elif opt in ("-n"):
            class0str = arg
        elif opt in ("-a"):
            class1str = arg
        elif opt in ("-M"):
            pars.noPrototypes = int(arg)
        elif opt in ("-D"):
            pars.winSize = int(arg)
        elif opt in ("-r"):
            numSimulations = int(arg)
        elif opt in ("-s"):
            mainSeed = int(arg)
        elif opt in ("-c"):
            changeType = arg

    try:
        experiment
    except NameError:
        print(helpMenu) 
        sys.exit()          

    if experiment=='AIDS':
        try:
            class0 = [class0str]
            if class0 == ['i']:
                class1 = ['a']
            else:
                class1 = ['i']
        except NameError:
            class0 = ['i']
            class1 = ['a']
        dataset = init_aids()

    elif experiment=='Mutagenicity':
        try:
            class0 = [class0str]
            if class0 == ['mutagen']:
                class1 = ['nonmutagen']
            else:
                class1 = ['mutagen']
        except NameError:
            class0 = ['nonmutagen']
            class1 = ['mutagen']
        dataset = init_mutagenicity()

    elif experiment=='Letter':
        try:
            class0 = [c for c in class0str]
        except NameError:
            class0 = ['A', 'E', 'F', 'H']
        try:
            class1 = [c for c in class1str]
        except NameError:
            class1 = ['F', 'H', 'I', 'K']
        dataset = init_letter()

    elif experiment[0:len('Markov15')] == 'Markov15':
        difficutly = int(experiment[len('Markov15')+1:])
        try:
            class0 = [class0str]
            if class0 == ['0']:
                class1 = ['1']
            else:
                class1 = ['0']
        except NameError:
            class0 = ['0']
            class1 = ['1']
        print("difficutly "+str(difficutly))
        dataset = init_markov15(difficutly)

    elif experiment == 'Delaunay':
        try:
            class0 = [class0str]
        except NameError:
            class0 = ['0']
        try:
            class1 = [class1str]
        except NameError:
            class1 = ['5']
        dataset = init_del()

    else:
        raise ValueError('experiment '+experiment+' not recognised...')


    # Set parameters

    pars.class0 = class0
    pars.class1 = class1

    # Output setup
    pars.nameNominal = ''
    for c in pars.class0:
        pars.nameNominal += c
    pars.nameNonnominal = ''
    for c in pars.class1:
        pars.nameNonnominal += c

    pars.folder = "%s_%s_p%dD%d_%s-%s" % (pars.timeCreation.strftime('%G%m%d_%H%M_%S'), \
        dataset.name, pars.noPrototypes, pars.winSize, pars.nameNominal, pars.nameNonnominal)

    pars.launchingCommand = launchingCommand


    pars = setting_toy(pars)


    if changeType == 'drift':
        pars.setDriftChange()
    elif changeType != 'abrupt':
        raise ValueError('type of change '+changeType+' not recognised...')

    
    dataset.loadDissimilarityMatrix()
    dataset.gmt_executable = gmt_executable

    if pars.noPrototypes > 0:
        simulation = simulations.SimulationTLC()
    else:
        pars.winSize = 1
        simulation = simulations.SimulationStdErr()

    simulation.set(pars, dataset, numSimulations)
    simulation.run(main_seed=mainSeed)




if __name__ == "__main__":
   main(sys.argv[1:])




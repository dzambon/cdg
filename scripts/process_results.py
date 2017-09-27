# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Load the raw results serialiseb by `pickle` and process them.
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
import pickle
from cdg import simulations

figureMerit = {}
figureMerit['dca']=['dca_rate', 'dca_rate_95ci'] 
figureMerit['arl']=['arl0', 'arl0_95ci', 'arl1', 'arl1_95ci' ] 
figureMerit['dcaarl']=figureMerit['dca']+figureMerit['arl'] 
figureMerit['mat']=['matlab'] 


helpMenu = 'Example:\n'
helpMenu += '\tpython3 process_result.py -i /absolute/path/to/experiment/folder\n'
helpMenu += '\n'
helpMenu += 'List of options:\n'
helpMenu += '-h\tprint this help\n'
helpMenu += '-i\tabsolute path to experiment folder\n'
helpMenu += '-l\ttipe of result\n'
helpMenu += '-r\tfile with results (simulation.pkl)\n'
for k, v in figureMerit.items():
    helpMenu += '\t\t'+k+': '+str(v)+'\n'



def main(argv):

    pathExp = None
    figMer = 'dca'
    fileRes = 'simulation.pkl'

    try:
        opts, args = getopt.getopt(argv,"i:l:r:h")
    except getopt.GetoptError:
        print(helpMenu)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMenu)
            sys.exit()
        elif opt in ("-i"):
            pathExp = arg
        elif opt in ("-l"):
            figMer = arg
        elif opt in ("-r"):
            fileRes = arg

    if pathExp is None:
        print(helpMenu)
        sys.exit(2)


    print('computing results...')
    pickleDict = pickle.load(open(pathExp+'/'+fileRes , "rb" ))
    sim = simulations.Simulation()  
    sim.savePickleDictionary(pickleDict)
    sim.pars.folder = pathExp

    try:
        return sim.processRawResults(figureMerit[figMer]), sim
    except: 
        print('something wrong... probably with the -l\n'+helpMenu)
        sys.exit(2)

if __name__ == "__main__":

    print('************************')
    print('* Process raw results  *')
    print('************************\n')

    tabularResult, _ = main(sys.argv[1:])

    for i in range(0,len(tabularResult[0])):
        print(tabularResult[0][i] + ':\t' + tabularResult[1][i])

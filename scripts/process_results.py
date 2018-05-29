# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Load .zip file containing the raw results.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 23/01/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import shutil
import getopt
import cdg.simulation.simulations
import zipfile

figureMerit = {}
figureMerit['dc'] = ['dc_rate', 'dc_rate_95ci']
figureMerit['dca'] = ['dca_rate', 'dca_rate_95ci']
figureMerit['arl'] = ['arl0', 'arl0_95ci', 'arl1', 'arl1_95ci']
figureMerit['fa1000'] = ['fa1000_rate', 'fa1000_rate_std']
figureMerit['dcaarl'] = figureMerit['dca'] + figureMerit['arl']
figureMerit['tnnls'] = figureMerit['dca'] + figureMerit['arl'] + figureMerit['fa1000']
figureMerit['mat'] = ['matlab']

helpMenu = 'Example:\n'
helpMenu += '\tpython3 process_result.py -i /absolute/path/to/experiment/folder\n'
helpMenu += '\n'
helpMenu += 'List of options:\n'
helpMenu += '-h\tprint this help\n'
helpMenu += '-i\tabsolute path to zipped folder\n'
helpMenu += '-l\ttipe of result\n'
helpMenu += '-r\tfile with results (simulation.pkl)\n'
for k, v in figureMerit.items():
    helpMenu += '\t\t' + k + ': ' + str(v) + '\n'


def main(argv):
    zippedExp = None
    figMer = 'dca'
    fileRes = 'simulation.pkl'

    try:
        opts, args = getopt.getopt(argv, "i:l:r:h")
    except getopt.GetoptError:
        print(helpMenu)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMenu)
            sys.exit()
        elif opt in ("-i"):
            zippedExp = arg
        elif opt in ("-l"):
            figMer = arg
        elif opt in ("-r"):
            fileRes = arg

    if zippedExp is None:
        print(helpMenu)
        sys.exit(2)

    print('computing results...')

    sim = cdg.simulation.simulations.SimulationOnline()
    zfile = zipfile.ZipFile(zippedExp)
    pathExp = zippedExp[:-4]
    print(zippedExp)
    print(pathExp)
    lastslash = pathExp.rfind('/')
    localPathExp = pathExp[lastslash + 1:] + "/"
    output_folder = "/tmp/cdg"
    unzipped_pickledfile = zfile.extract(localPathExp + fileRes, path=output_folder)
    sim.deserialise(unzipped_pickledfile)
    shutil.rmtree(output_folder)

    try:
        return sim.process_results(figureMerit[figMer]), sim
    except:
        print('something wrong... probably with the -l\n' + helpMenu)
        sys.exit(2)


if __name__ == "__main__":

    print('************************')
    print('* Process raw results  *')
    print('************************\n')

    tabularResult, _ = main(sys.argv[1:])

    for i in range(0, len(tabularResult[0][0])):
        print(tabularResult[0][0][i] + ':\t' + tabularResult[0][1][i])

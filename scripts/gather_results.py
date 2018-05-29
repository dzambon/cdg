# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# Collect several results in a single table.
#
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 29/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import argparse
import process_results

# Opening text
opening_text = '***************************\n'
opening_text += '* Collect results         *\n'
opening_text += '***************************\n'
opening_text += 'Author: Daniele Zambon\n'
opening_text += 'eMail: daniele.zambon@usi.ch\n'

possibleExp = []
possibleExp.append(('AE_FH', 'L-D2'))
possibleExp.append(('AEFHI_KLMNT', 'L-D5'))
possibleExp.append(('AEFH_FHIK', 'L-O'))
possibleExp.append(('AEFHI_FHI', 'L-S'))
possibleExp.append(('AEFHI_FHI', 'L-S'))
possibleExp.append(("nonmutagen_mutagen", 'MUT'))
possibleExp.append(("i_a", 'AIDS'))
possibleExp.append(("Markov", 'M'))
possibleExp.append(("Delaunay", 'D'))

possibleManifold = []
possibleManifold.append(('E(', ' E  \t'))
possibleManifold.append(('S(', ' S  \t'))
possibleManifold.append(('H(', ' H  \t'))
possibleManifold.append(('DR(', ' DR \t'))
possibleManifold.append(('FrMu(', 'FrMu\t'))
possibleManifold.append(("SpectralGap", ' SG \t'))
possibleManifold.append(("Density", 'den \t'))
possibleManifold.append(("GraphMean", ' GM \t'))

helpMenu = 'List of figures of merit available:\n'
for k, v in process_results.figureMerit.items():
    helpMenu += '\t\t' + k + ': ' + str(v) + '\n'
parser = argparse.ArgumentParser(description=helpMenu)
parser.add_argument('-f', '--filename', type=str,
                    default='experiment_list.txt',
                    help='file containing the list of zipped experiment results')
parser.add_argument('-l', '--latex', type=str,
                    default='dca',
                    help='format of the LaTeX table entry'+helpMenu)


def main(argv):
    print(opening_text)

    # Read arguments
    args = parser.parse_args()

    # Parse result file
    print("reading %s ..." % args.filename)
    listExp = []
    try:
        with open(args.filename, 'r') as f:
            for line in f:
                line = line.strip()
                # print(line)
                if len(line) > 1:
                    # print('*'+str(len(line)))
                    listExp.append(line)
                    # print(line)
    except FileNotFoundError:
        print('FileNotFoundError: ' + args.filename)
        print(parser.print_help())
        # print(helpMenu)
        sys.exit(2)

    # Arrange results
    print('retrieving results...')
    table = []
    for exp in listExp:

        tabularResult, simulation = process_results.main(['-i', exp, '-l', args.latex])
        output = ''
        header = ''
        # for i in range(0,len(tabularResult[0])):
        #     header += ' & ' + tabularResult[0][i] + '\t'
        #     output += ' & ' + tabularResult[1][i] + '\t'
        dataset_str = None

        for i in range(0, len(tabularResult[0][0])):
            header += ' & ' + tabularResult[0][0][i] + '\t'
            output += ' & ' + tabularResult[0][1][i] + '\t'
            # print(tabularResult[0][0][i] + ':\t' + tabularResult[0][1][i])

        man_str = "..."
        for man in possibleManifold:
            if man[0] in exp:
                man_str = man[1]
                break

        for nameExp in possibleExp:
            if nameExp[0] in exp:
                if 'Delaunay' in exp:
                    dataset_str = nameExp[1] + '-' + exp.split('_')[-1].split('.')[0]
                elif 'Markov' in exp:
                    dataset_str = nameExp[1] + '-' + exp.split('_')[-3]
                else:
                    dataset_str = nameExp[1]

                break

        try:
            manDim = simulation.pars.manifold.manifold_dimension
        except:
            manDim = -1
        try:
            noPro = simulation.pars.manifold.no_prototypes
        except:
            noPro = -1
        noWin = simulation.pars.window_size

        # todo fix this
        if noPro is None:
            noPro = -1
        if manDim is None:
            manDim = -1
        print("fix this")

        tableEntry = "%s & %s\t & %d & %d & %d %s \\\\" % \
                     (man_str, dataset_str, noPro, manDim, noWin, output)
        header =    "manifold &  dataset &  M    & d     & n & " + header

        table.append((nameExp[1], manDim, noWin, tableEntry))

    print('sorting table entries...')
    table = sorted(table, key=lambda tup: (tup[0], tup[1], tup[2]))

    # Print results
    # print('\t...\t' + header)
    print(header + " \\\\")
    print("\\hline")
    for tableEntry in table:
        print(tableEntry[3])


if __name__ == "__main__":
    main(sys.argv[1:])

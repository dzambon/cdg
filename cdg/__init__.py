# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
# --------------------------------------------------------------------------------
import cdg

__version__ = '2.1'

def get_git_commit():
    import subprocess
    command = "git --git-dir " + cdg.__path__[0][:-4] + "/.git rev-parse HEAD"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    cdg_git_commit_hash, err = process.communicate()
    return cdg_git_commit_hash, err

from cdg.graph.graph import Graph, DataSet, GraphMeasure, Kernel, Distance
from cdg.embedding.embedding import Embedding
from cdg.changedetection.changedetection import ChangeDetectionTest, ChangePointMethod
from cdg.simulation.simulations import SimulationCDT, SimulationCPM
from cdg.simulation.parameters import ParametersCDT, ParametersCPM


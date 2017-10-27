# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Description:
# ---------
# This is intended as a python interface for dealing with the IAM Graph Database
# repository [1], Delaunay Graphs [2] and for interacting with the 
# GraphMatchingToolkit [3].
# It support also the Dealunay Graph Database (by Daniele Zambon).
#
#
# References:
# ---------
# [1] Riesen, Kaspar, and Horst Bunke. "IAM graph database repository for graph 
#   based pattern recognition and machine learning." Structural, Syntactic, and 
#   Statistical Pattern Recognition (2008): 287-297.
#
# [2] Zambon, Daniele, Livi, Lorenzo and Alippi, Cesare. "Detecting Changes in 
#   Sequences of Attributed Graphs." IEEE SSCI (2017).
#
# [3] K. Riesen, S. Emmenegger and H. Bunke. A Novel Software Toolkit for Graph
#   Edit Distance Computation. In W.G. Kropatsch et al., editors, Proc. 9th Int.
#   Workshop on Graph Based Representations in Pattern Recognition,
#   LNCS 7877, 142–151, 2013.
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
# Last Update: 27/10/2017
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import xml.etree.ElementTree as et
import numpy as np
import random
import subprocess
import datetime
import pickle
import os
import sys
from cdg import graph



class Database:

    precomputedDissimilarityMatrix = False
    filenameGraphNameMap = "graph_name.map"
    filenameDissimilarityMatrix = "dissimilarity_matrix.dat"
    pickleDissimilarityMatrix = "dissimilarity_matrix.pkl"
    gmt_executable = "../../GMT/dz_GraphMatching.jar"


    def __init__(self,path):
        self.path = path

        self.graphNameAndClass = []
        self.classes = []
        self.elements = {}

        self.name = 'Generic'
        self.notes = None
        
        if not os.path.isdir(self.path):
            print('Warning: ' + self.path+ ' does not exists')
        else:
            foundGxl=False
            for fname in os.listdir(self.path):
                if fname.endswith('.gxl'):
                    foundGxl = True
                    break
            if not foundGxl and os.path.isdir(self.path):
                 print('Warning: no .gxl in path: ' + self.path)



    def __str__(self):
        string = ''
        selectedAttributes = ['gmt_executable','path','name','notes','classes']
        # for key in self.__dict__.keys():
        for key in selectedAttributes:
            string += "%s = %s\n" % (key,self.__dict__[key])

        return string

    def loadGraphNameMap(self):
        ct = 0
        if os.path.isfile(self.path+"/"+self.filenameGraphNameMap):
            self.graphNameAndClass = []
            with open(self.path + "/" + self.filenameGraphNameMap, 'r') as f:
                for line in f:
                    if line[0] != '#':
                        tokens = line.split(',')
                        self.graphNameAndClass.append((int(tokens[0]),tokens[1],tokens[2]))
                        ct +=1
            self.noGraphs = ct
        else:
            self.generateGraphNameMap()

        self.classes = []
        self.elements = {}
        for g in self.graphNameAndClass:
            found = False
            for c in self.classes:
                if g[2] == c:
                    found = True
                    self.elements[c].append(int(g[0]))
                    break
            if not found:
                self.classes.append(g[2])
                self.elements[g[2]] = [g[0]]


    def generateGraphNameMap(self):

        # numGraphs = find_class(self.path)
        graphNameList = []
        for file in os.listdir(self.path):
            if file.endswith(".gxl"):
                graphNameList.append(file)
        graphNameList.sort()

        self.graphNameAndClass = []
        ct=0
        for graph in graphNameList:
            type = Database.find_class(self.path, graph)
            self.graphNameAndClass.append( (ct, graph, type) )
            ct +=1   
            sys.stdout.write("\r" + str(ct))
            sys.stdout.flush()
        self.saveGraphNameMap()


    @classmethod
    def find_class(cls, db_path, name=None):

        files = [db_path + "/train.cxl", db_path + "/test.cxl", db_path + "/valid.cxl", db_path + "/validation.cxl"]
        for file in files:
            try:
                tree = et.parse(file)
                entries = tree.findall(".//print")
                for e in entries:
                    if e.attrib['file'] == name:
                        return e.attrib['class']
            except FileNotFoundError:
                tmp = 0

        raise ValueError("class of %s not found" % name)


    def saveGraphNameMap(self):
        # write graph_name.map
        f = open(self.path+"/"+self.filenameGraphNameMap, 'w')
        # f.write("# Author: Daniele Zambon\n")
        # f.write("# eMail: daniele.zambon@usi.ch\n")
        f.write("# Generated: %s" % datetime.datetime.now().strftime('%G/%m/%d %H:%M'))
        f.write("\n# Header description: incremental ID, filename of the graph, class,")
        ct=0
        for graphEntry in self.graphNameAndClass:
            f.write("\n%d,%s,%s," % graphEntry )
            ct +=1

        self.noGraphs = ct
        f.close()








    def loadDissimilarityMatrix(self, ):
        self.gmt = GMT(self.gmt_executable)

        self.loadGraphNameMap()

        # print('ini dissimilarity matrix...')
        if os.path.isfile(self.path+"/"+self.pickleDissimilarityMatrix):

            # Load from pickle file.
            dissMatAndInfo = pickle.load( open( self.path+"/"+ self.pickleDissimilarityMatrix, "rb" ) )
            self.dissimilarityMatrix = dissMatAndInfo['dissMat']

            if self.precomputedDissimilarityMatrix and np.min(self.dissimilarityMatrix)<0 :
                raise ValueError("dz: you are trying to load the dissimilarity matrix as precomputed, but it is not fully completed yet.")


        else:
            self.generateDissimilarityMatrix()



    def generateDissimilarityMatrix(self):

        if self.precomputedDissimilarityMatrix:
            self.precomp_split()
        else:
            # init empty matrix
            # self.loadGraphNameMap()
            print('init dissimilarity matrix...')
            self.saveDissimilarityMatrix(dissimilarityMatrix=-np.ones((len(self.graphNameAndClass),len(self.graphNameAndClass))))


    def saveDissimilarityMatrix(self,dissimilarityMatrix=None):
        if dissimilarityMatrix is not None:
            self.dissimilarityMatrix = dissimilarityMatrix.copy()
            noGraphs = dissimilarityMatrix.shape[0]
            if noGraphs != dissimilarityMatrix.shape[1]:
                print(dissimilarityMatrix.shape)
                raise ValueError('Something wrong with the matrix')

        noGraphs = self.dissimilarityMatrix.shape[1]
        dissMatAndInfo = {'dissMat': self.dissimilarityMatrix, 'generated': datetime.datetime.now().strftime('%G/%m/%d %H:%M')}

        # Save as pickle file.
        pickle.dump(dissMatAndInfo, open(self.path+"/"+ self.pickleDissimilarityMatrix, "wb" ) )

        # Save as txt file.
        f = open(self.path+"/"+self.filenameDissimilarityMatrix, 'w')
        f.write("# Generated: %s\n" % dissMatAndInfo['generated'])
        f.write("# Num of elements: %d\n" % noGraphs)
        f.write("# ... deprecated\n")
        # for raw in self.dissimilarityMatrix:
        #     for c in raw:
        #         f.write("%.3f,\n" % c )
        f.close()


    def precomp_rows(self, firstIncluded,lastExcluded):

        return self.compute_submatrix([i for i in range(firstIncluded,lastExcluded)], [i for i in range(0,len(self.graphNameAndClass))])





    def compute_submatrix(self, rows, columns):

        propertyDict = self.getGMTProperties()

        for k in propertyDict:
            self.gmt.property[k]=propertyDict[k]

        # # Auxiliary files for the computation

        # bypass concurrency conflicts
        conflictId = datetime.datetime.now().strftime('%G%m%d%H%M%S') + '_' + str(random.randint(0, 10000))
        # lists of graphs
        raw_xml = conflictId + "raw.gmt"
        column_xml = conflictId + "column.gmt"
        result = conflictId + "result.gmt"
        # parameters of the GED
        gmt_parameters = conflictId + "gmt_parameters.gmt"

        # # Generate XML Datasets

        f_r = open(raw_xml, 'w')
        f_c = open(column_xml, 'w')
        f_r.write(GMT.getCxlOpening())
        f_c.write(GMT.getCxlOpening())
        numOfRows = 0
        numOfColumns = 0
        t = 0

        for graphName in self.graphNameAndClass:

            if t in rows:
                f_r.write(GMT.getCxlGraphElement(graphName[1], graphName[2]))
                numOfRows += 1
            if t in columns:
                f_c.write(GMT.getCxlGraphElement(graphName[1], graphName[2]))
                numOfColumns += 1

            t += 1

        f_r.write(GMT.getCxlClosing())
        f_c.write(GMT.getCxlClosing())
        f_r.close()
        f_c.close()

        # # Compute matrix



        # compute dissimilarities
        self.gmt.property_file = gmt_parameters
        self.gmt.set_input_graphsets(raw_xml, numOfRows, column_xml, numOfColumns)
        self.gmt.set_database_path(self.path + "/")
        self.gmt.set_result_file(result)
        dissimilarityMatrix = self.gmt.launch(use_precomputed_result=False, displayGMTOutput=True)

        # # Clean useless file

        # remove unnecessary files
        os.remove(raw_xml)
        os.remove(column_xml)
        os.remove(result)
        os.remove(gmt_parameters)

        return dissimilarityMatrix



    def precomp_split(self):
        

        # load filename
        self.loadGraphNameMap()

        dimBatch = 10
        numBatches = self.noGraphs//dimBatch
        if numBatches * dimBatch < self.noGraphs:
            numBatches += 1
        print(numBatches)            

        dissimilarityMatrix = np.zeros((0,self.noGraphs))
        print(dissimilarityMatrix.shape)            
        for b in range(0,numBatches):
            firstIncluded = b*dimBatch
            lastExcluded  = min([self.noGraphs, firstIncluded + dimBatch])
            # if firstIncluded==lastExcluded:
            #    break
            print("%d)%d __ %d" % (b,firstIncluded,lastExcluded) ) 
            dissMatrix_tmp = self.precomp_rows(firstIncluded,lastExcluded)
            dissimilarityMatrix = np.concatenate((dissimilarityMatrix,dissMatrix_tmp), axis=0)
            print(dissimilarityMatrix.shape)

        self.saveDissimilarityMatrix(dissimilarityMatrix)




    def compute_single_entry(self, input_a,input_b):
        

        # # Auxiliary files for the computation

        # bypass concurrency conflicts
        conflictId = datetime.datetime.now().strftime('%G%m%d%H%M%S')+'_'+ str(random.randint(0,10000))
        # lists of graphs
        raw_xml = conflictId + "raw.gmt"
        column_xml = conflictId + "column.gmt"
        result = conflictId + "result.gmt"
        # parameters of the GED
        gmt_parameters = conflictId + "gmt_parameters.gmt"

        # # Generate XML Datasets

        f_r = open(raw_xml, 'w')
        f_c = open(column_xml, 'w')
        f_r.write(GMT.getCxlOpening())
        f_c.write(GMT.getCxlOpening())

        el = self.getNameAndClass(input_a)
        f_r.write(GMT.getCxlGraphElement(el[1],el[2]))
        el = self.getNameAndClass(input_b)
        f_c.write(GMT.getCxlGraphElement(el[1],el[2]))

        f_r.write(GMT.getCxlClosing())
        f_c.write(GMT.getCxlClosing())
        f_r.close()
        f_c.close()
         

        # # Compute matrix

       

        # compute dissimilarities
        self.gmt.property_file = gmt_parameters
        self.gmt.set_input_graphsets(raw_xml, 1, column_xml, 1)
        self.gmt.set_database_path(self.path+"/")
        self.gmt.set_result_file(result)
        dissMat = self.gmt.launch(use_precomputed_result=False, displayGMTOutput=True)


        # # Clean useless file

        # remove unnecessary files
        os.remove(raw_xml)
        os.remove(column_xml)
        os.remove(result)
        os.remove(gmt_parameters)


        return dissMat[0,0]




    def getNameAndClass(self,id):
        for el in self.graphNameAndClass:
            if el[0] == id:
                return el
        return None


    def subDissimilarityMatrix(self,rows,columns):

        if type(rows) is int:
            rows = [rows]
        if type(columns) is int:
            columns = [columns]
        subMatrix = - np.ones((len(rows), len(columns)))


        if not self.precomputedDissimilarityMatrix:

            for i in range(0,len(rows)):
               for j in range(0,len(columns)):
                    if self.dissimilarityMatrix[ rows[i] ][ columns[j] ]>=0:
                        subMatrix[i,j]=0

            computedSomething = False
            for i in range(0, len(rows)):
                if sum(subMatrix[i, :]) < 0:
                    cols = []
                    jcols = []
                    for j in range(0, len(columns)):
                        if subMatrix[i,j]<0:
                            cols.append(columns[j])
                            jcols.append(j)
                    if len(cols)>0:
                        cols = list(set(cols))
                        newDissMat = self.compute_submatrix([rows[i]], cols)
                        for j in range(0, len(cols)):
                            self.dissimilarityMatrix[rows[i]][cols[j]] = newDissMat[0,j]
                            subMatrix[i,jcols[j]]=0
                            computedSomething = True
                        print('sparsity: %d/%d' % (-sum(sum(subMatrix)),len(rows)*len(columns)) )

            if computedSomething:
                self.saveDissimilarityMatrix()

        for i in range(0,len(rows)):
           for j in range(0,len(columns)):
                subMatrix[i,j] = self.dissimilarityMatrix[ rows[i] ][ columns[j] ]

        return subMatrix


    def generateBootstrappedStream(self, classes, length, prc=None):

        if prc is None:

            bin = [] #[1./len(classes) for c in classes]
            for i in range(0,len(classes)):
                bin.append( (i+1.)/len(classes) )

            binStream = []
            for t in range(0,length):
                binStream.append(bin)

        elif len(prc) == len(classes):

            bin = [prc[0]] #[1./len(classes) for c in classes]
            for i in range(1,len(prc)):
                bin.append( bin[-1] + prc[i])
            if bin[-1]<0.99:
                raise ValueError("prc do not sum to 1")
            else:
                bin[-1]=1

            binStream = []
            for t in range(0,length):
                binStream.append(bin)

        elif len(prc) == length:
            binStream = []
            for prc_t in prc:
                bin = [prc_t[0]]  # [1./len(classes) for c in classes]
                for i in range(1, len(prc_t)):
                    bin.append(bin[-1] + prc_t[i])
                if bin[-1]<0.99:
                    raise ValueError("prc do not sum to 1")
                else:
                    bin[-1]=1

                binStream.append(bin)

        else:
            raise ValueError("length mismatch")


        if type(prc) is list and len(prc)==length:
            print('bootstrapping with drift (probably)...')
        else:
            print('bootstrapping (prc = %s)...' % prc)

        elements = []
        for c in classes:
            elements += self.elements[c]

        stream_tmp = []
        for i in range(0,len(classes)):
            stream_tmp.append(np.random.choice(self.elements[classes[i]], length, True))


        r = np.random.rand(length)
        cc =[]
        for t in range(0,length):
            i=0
            while r[t] > binStream[t][i]:
                i +=1
            cc.append(i)

        stream = []
        for t in range(0,length):
            stream.append(stream_tmp[cc[t]][t])
        return stream





class IAMGeometric(Database):

    def __init__(self,path):
        Database.__init__(self,path)
        self.name = 'Geometric'

        
    def getGMTProperties(self):

        propertyDict={}

        propertyDict['numOfNodeAttr'] = 2
        propertyDict['nodeAttr0'] = 'x'
        propertyDict['nodeAttr1'] = 'y'

        propertyDict['nodeCostType0'] = 'squared'
        propertyDict['nodeCostType1'] = 'squared'

        propertyDict['nodeAttr0Importance'] = 1.0
        propertyDict['nodeAttr1Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.5

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0


        return propertyDict



class IAMMolecule(Database):

    def __init__(self,path):
        Database.__init__(self,path)
        self.name = 'Molecule'

    def getGMTProperties(self):

        propertyDict={}

        propertyDict['numOfNodeAttr'] = 1
        propertyDict['nodeAttr0'] = 'chem'

        propertyDict['nodeCostType0'] = 'sed'
        propertyDict['nodeAttr0Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 1
        propertyDict['edgeAttr0'] = 'valence'
        propertyDict['edgeCostType0'] = 'squared'
        propertyDict['edgeAttr0Importance'] = 1.0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.5

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict







class Letter(IAMGeometric):

    def __init__(self,main_path,distortion):
        IAMGeometric.__init__(self,main_path+"/"+distortion)
        self.main_path=main_path
        self.distortion=distortion
        self.name = 'Letter'
        self.notes = 'distortion = ' + str(distortion) 




class Mutagenicity(IAMMolecule):
    def __init__(self,path):
        IAMMolecule.__init__(self,path)
        self.name = 'Mutagenicity'


class AIDS(IAMMolecule):

    def __init__(self,path):
        IAMMolecule.__init__(self,path)
        self.name = 'AIDS'

    def getGMTProperties(self):

        propertyDict = IAMMolecule.getGMTProperties(self)
        propertyDict['nodeAttr0'] = 'symbol'

        return propertyDict





class Delaunay(Database):


    def __init__(self, path):

        Database.__init__(self, path)

        self.name = 'Delaunay'
        self.noClasses=20

    def generateNewDataset(self, seedPoints=10, noGraphs=10, seed=None):

        if seed is not None:
            random.seed(seed)

        radius = 10.
        if type(seedPoints) is int:
            seedPoints= np.random.rand(seedPoints, 2) * radius

        if os.path.isfile(self.path+"/" + self.filenameGraphNameMap):
            os.remove(self.path+"/" + self.filenameGraphNameMap)
        if os.path.isfile(self.path+"/" + self.filenameDissimilarityMatrix):
            os.remove(self.path+"/" + self.filenameDissimilarityMatrix)
        if os.path.isfile(self.path+"/" + self.pickleDissimilarityMatrix):
            os.remove(self.path+"/" + self.pickleDissimilarityMatrix)


        for c in range(0,self.noClasses+1):


            if c==0:
                points=seedPoints
            else:
                points = np.zeros((seedPoints.shape[0],seedPoints.shape[1]))
                for i in range(0,seedPoints.shape[0]):
                    points[i, 0] = seedPoints[i, 0] + np.sin(np.random.rand(1) * 2. * np.pi)*radius
                    points[i, 1] = seedPoints[i, 1] + np.cos(np.random.rand(1) * 2. * np.pi)*radius
                radius *= .66

            delGen = graph.DelaunayGenerator(path=self.path+'/'+str(c), classID=c)
            delGen.setFundamentalPoints(points=points)
            sigma = 1
            for i in range(0,noGraphs):
                delGen.generateNewGraph(sigma=sigma, counter=i)



    def generateGraphNameMap(self):

        # numGraphs = find_class(self.path)
        graphNameList = []
        # for file in os.listdir(self.path+"/Training")+os.listdir(self.path+"/Validation")+os.listdir(self.path+"/Test"):
        for pathAbs, s, filesAbs in os.walk(self.path):
            for fileAbs in filesAbs:
                if fileAbs.endswith(".gxl"):
                    pathRel = os.path.relpath(pathAbs, self.path)
                    fileRel = os.path.join(pathRel, fileAbs)
                    # print(fileRel)
                    graphNameList.append(fileRel)
                sys.stdout.write(".")
                sys.stdout.flush()
        graphNameList.sort()
        
        self.graphNameAndClass = []
        ct=0
        for graph in graphNameList:
            className = graph.split("_")[1]
            self.graphNameAndClass.append( (ct, graph, className) )
            ct +=1
            print((ct, graph, className) )

        Database.saveGraphNameMap(self)


    def getGMTProperties(self):

        propertyDict = {}

        propertyDict['node'] = 1.0
        propertyDict['edge'] = .1

        propertyDict['numOfNodeAttr'] = 1
        propertyDict['nodeAttr0'] = 'weight'
        propertyDict['nodeCostType0'] = 'csvDouble'
        propertyDict['nodeAttr0Importance'] = 1.0

        propertyDict['multiplyNodeCosts'] = 0
        propertyDict['pNode'] = 2

        propertyDict['undirected'] = 1

        propertyDict['numOfEdgeAttr'] = 0

        propertyDict['multiplyEdgeCosts'] = 0
        propertyDict['pEdge'] = 1

        propertyDict['alpha'] = 0.7

        propertyDict['outputGraphs'] = 0
        propertyDict['outputEditpath'] = 0
        propertyDict['outputCostMatrix'] = 0
        propertyDict['outputMatching'] = 0

        propertyDict['simKernel'] = 0

        return propertyDict


    def renderSomeExamples(self, plt, noGraphs=3, classID='0', offset = None):

        if offset is None:
            offset = 14

        ids = np.random.choice(self.elements[classID],noGraphs)

        for ct in range(0,noGraphs):
            gxlfile = self.path + '/'+self.getNameAndClass(ids[ct])[1]
            if gxlfile[0] != '/':
                gxlfile = os.getcwd() + '/' + gxlfile
            g = graph.Graph(gxlfile)

            g.load()

            for i in range(0, g.n):
                for j in range(0, g.n):
                    if g.adj[i][j]>0:
                        plt.plot([g.xpos[i]+offset *ct,g.xpos[j]+offset *ct], [g.ypos[i],g.ypos[j]],color='#505050', linewidth=1)
                        plt.plot([g.xpos[i]+offset *ct,g.xpos[j]+offset *ct], [g.ypos[i],g.ypos[j]],color='#007acc', marker='o', markersize=3.8, linestyle='')


        plt.axis([-4, offset *noGraphs+2, -4, 14])




def dot2gxl(folder):
    graphNameList = []
    ct=0
    for filedot in os.listdir(folder):
        if filedot.endswith(".dot"):
            filegxl = filedot[:-4]+".gxl"

            command = "dot2gxl -g " + folder+"/"+filedot
            print("executing: " + command)
            # out=subprocess.Popen(command.split(), stdout=subprocess.PIPE).wait()
            out=subprocess.Popen(command.split(), stdout=open(folder+"/"+filegxl, 'wb')).wait()
            print(out)
            ct += 1
    return ct




class DelaunayGenerator():
    
    def __init__(self, path='.', classID=42, nameFamily=None):
        self.classID = classID
        # self.nameGraph = nameGraph
        self.path = path
        if nameFamily is None:
            self.nameFamily = 'del_'+str(classID)
        else:
            self.nameFamily = nameFamily

    def setFundamentalPoints(self, points):
        self.points = points
        if points.shape[1]!=2:
            raise ValueError("dimension must be two")
        self.noVertices = points.shape[0]

    def generateFundamentalPoints(self, noVertices):
        self.setFundamentalPoints(self, np.random.randn(noVertices, 2))
        
    def generateNewGraph(self, sigma=.2, radius=1, counter=0):
        newPoints = self.points + np.random.randn(self.noVertices,2)*sigma
        newPoints *= radius
        newAdjMat = DelaunayGenerator.generate_delaunay_adjacency(newPoints)

        self.save_GXL(newPoints,newAdjMat, self.nameFamily+'_'+str(counter))


    def save_GXL(self,newPoints,newAdjMat, nameGraph):

        f = open(self.path+'/'+nameGraph+'.gxl','w')

        f.writelines('<?xml version="1.0" encoding="UTF-8"?>\n')
        # f.writelines('<!DOCTYPE gxl SYSTEM "http://www.gupro.de/GXL/gxl-1.0.dtd">\n')
        # f.writelines('<gxl xmlns:xlink=" http://www.w3.org/1999/xlink">\n')
        f.writelines('<gxl>\n')
        f.writelines('\t<graph id="%s" edgeids="false" edgemode="undirected">\n' % nameGraph)
        f.writelines('\t\t<attr name="classid" kind="graph">\n')
        f.writelines('\t\t\t<string>%s</string>\n' % self.classID)
        f.writelines('\t\t</attr>\n')


        for p in range(0,newPoints.shape[0]):
            f.writelines('\t\t<node id="v_%d">\n' % p )
            f.writelines('\t\t\t<attr name="weight">\n')
            f.writelines('\t\t\t\t<string>%s</string>\n' % DelaunayGenerator.weightVectorString(newPoints[p,:]) )
            f.writelines('\t\t\t</attr>\n')
            f.writelines('\t\t</node>\n')

        for p1 in range(0,newPoints.shape[0]):
            for p2 in range(p1+1, newPoints.shape[0]):
                if newAdjMat[p1,p2]==1:
                    f.writelines('\t\t<edge from="v_%d" to="v_%d" isdirected="false"></edge>\n' % (p1,p2) )

        f.writelines('\t</graph>\n</gxl>\n')
        f.close()

        return Graph(filename = nameGraph+'.gxl', directory=self.path)



    @classmethod
    def generate_delaunay_adjacency(cls,points):
        noVertices = points.shape[0]

        tri = scp.Delaunay(points)
        adjMatrix = np.zeros((noVertices, noVertices))

        # create adjacency matrix
        for t in tri.simplices:
            for i in range(0, 3):
                j = np.mod(i + 1, 3)
                adjMatrix[t[i], t[j]] = 1
                adjMatrix[t[j], t[i]] = 1

        return adjMatrix #, degrees

    @classmethod
    def weightVectorString(cls,point):
        string = "["
        for e in point:
            string += str(e) + ","

        string = string[:-1] + "]"
        return string






class GMT:
    """
    The GMT class is a wrapper of the GraphMatchingToolkit (GMT).
    The standard routine can be:
        # Setup the GMT Instance
        gmt = GMT('../../GraphMatchingToolkit/dz_GraphMatching.jar')
        gmt.set_database_path('../../_Graph_Database_perturbed/Letter/LOW/')
        gmt.set_input_graphs('prototypes.xml', n_p, 'target.xml', n_t)
        gmt.set_result_file('result.txt')
        # Launch GMT Procedure
        dissimilarity_matrix = gmt.launch()
    """

    def __init__(self,gmt_executable,path=None,jar=True):
        """
        Defines which is the command to launch the GMT.
        """
        self.gmt_executable=gmt_executable
        if jar:
            self.command = "java -jar " + self.gmt_executable + " "
        else:
            self.command = "java -cp " + self.gmt_executable + " algorithm.GraphMatching "

        self.n_source = 0
        self.n_target = 0

        self.property = {}

        self.property_file = './dz_graph_GMT_propertyfile_default.prop'

        self.property['matching'] = 'VJ'
        self.property['s'] = ''
        self.property['adj'] = 'best'

        self.property['node'] = 1.0
        self.property['edge'] = 1.0

        self.property['result'] = './dz_graph_GMT_resultfile_default.txt'
        self.property['path'] = path

    def set_input_graphsets(self,source, n_source, target, n_target):
        self.property['source'] = source
        self.n_source = n_source
        self.property['target'] = target
        self.n_target = n_target

    def set_database_path(self,path):
        self.property['path'] = path

    def set_result_file(self,result):
        self.property['result'] = result

    def create_property_file(self):
        f = open(self.property_file, 'w')
        for key in self.property.keys():
            f.writelines(key + " = " + str(self.property[key]) + '\n')
        f.close()

 


    def launch(self, gxl_source=None, gxl_target=None, use_precomputed_result=False, displayGMTOutput=True):
        """
        Actually launches the GMT with the `parameters` specified, and arrange the results in a
        bidimensional array whose entries d_{ij} = dissimilarity(source_i,target_j).
        """
        if use_precomputed_result:
            print("dz:Warning: I'm reading the result file already available, I'm not recomputing it.")
        else:

            if gxl_source is not None and gxl_target is not None:
                self.property['singlePair'] = 1
                self.n_source = 1
                self.n_target = 1

            self.create_property_file()
            command = self.command + self.property_file

            if gxl_source is not None and gxl_target is not None:
                command = command + ' ' + gxl_source + ' ' + gxl_target

            print("executing: " + command)
            if displayGMTOutput:
                subprocess.Popen(command.split()).wait()
            else:
                subprocess.Popen(command.split(), stdout = subprocess.PIPE).wait()

        # reimport results
        f = open(self.property['result'], 'r')
        while True:
            line = f.readline()
            # print(line)
            if line[0:1] != '#':
                break
        dist = np.zeros((self.n_source, self.n_target))
        r = 0
        c = 0
        while True:
            # print(r,"/",c)
            try:
                dist[r][c] = float(line)
            except ValueError:
                print(line)
                raise ValueError(line)
            c += 1
            if c >= self.n_target:
                c = 0
                r += 1
            if r >= self.n_source:
                break
            # print(line)
            line = f.readline()

        return dist

    @classmethod
    def getCxlOpening(cls):
        return '<?xml version="1.0"?>\n<GraphCollection>\n<graphs>\n'

    @classmethod
    def getCxlClosing(cls):
        return '</graphs>\n</GraphCollection>\n'

    @classmethod
    def getCxlGraphElement(cls, graphName, graphClass):
        return '<print file="'+graphName+'" class="' + graphClass + '"/>\n'


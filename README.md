Author: *Daniele Zambon*,  
Affiliation: *Università della Svizzera italiana*  
eMail: `daniele.zambon@usi.ch`  
Last Update: *18/05/2018*   


# Concept Drift and Anomaly Detection in a Sequence of Graphs

The cdg module is the reference code for most of my publications. 
Please find a detailed description of the procedures these: [tnnls17], [ijcnn18] and [ssci17]. Some demostrations are also available.


## The module

The code is written in `python 3`. 
In the module you will find following folders
* **`cdg/graph`** interface for datasets of graphs and dissimilarities.
* **`cdg/embedding`** several types of numeric representations of graphs, such as, dissimilarity representation and manifold embeddings [tnnls17, ijcnn18]. 
* **`cdg/changedetection`** tests for concept drift (change) detection [tnnls17].
* **`cdg/util`** utilities for the module.
* **`cdg/simulation`** simulates a sequence of graphs and applies detection tests.

You will then find some scripts (`scritps`), e.g., to aggregate results, and code snippets (`demo`).


## Requirements 

You may need the `graph-matching-toolkit` [1] (available [here](https://github.com/dan-zam/graph-matching-toolkit)) and `graphviz` [2].
A good place to locate the `graph-matching-toolkit` is `./graph-matching-toolkit/graph-matching-toolkit.jar`.

## Installation

Go to your preferred directory, then 
```bash
git clone https://github.com/dan-zam/cdg
cd cdg
sudo pip3 install -e .
```

## References

[tnnls17] `bib/zambon_tnnls17.bib`  
  Zambon, Daniele, Cesare Alippi, and Lorenzo Livi.  
  Concept Drift and Anomaly Detection in Graph Streams.  
  IEEE Transactions on Neural Networks and Learning Systems (2018).  

[ssci17] `bib/zambon_ssci17.bib`  
  Zambon, Daniele, Livi, Lorenzo and Alippi, Cesare.   
  Detecting Changes in Sequences of Attributed Graphs.  
  IEEE Symposium Series in Computational Intelligence (2017).  

[ijcnn18] `bib/zambon_ijcnn18.bib`  
  Zambon, Daniele, Livi, Lorenzo and Alippi, Cesare.  
  Anomaly and Change Detection in Graph Streams through Constant-Curvature Manifold Embeddings.  
  IEEE International Joint Conference on Neural Networks (2018).  

[1]  
  Riesen, Kaspar, Sandro Emmenegger, and Horst Bunke.   
  A novel software toolkit for graph edit distance computation.  
  International Workshop on Graph-Based Representations in Pattern   Recognition. 2013.

[2]   
http://www.graphviz.org   


## Licence

Licence: BSD-3-Clause

Copyright (c) 2017-2018, Daniele Zambon
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


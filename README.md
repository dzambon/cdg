
<img src="logo.svg" alt="logo cdg" width="160"/>


# Change Detection in a sequence of Graphs

This package is the reference code for most of my publications, please consider citing them.

Some are the following

* [Change-Point Methods on a Sequence of Graphs](https://dzambon.github.com/publications/zambon2019change)
* [Concept Drift and Anomaly Detection in Graph Streams](https://dzambon.github.com/publications/zambon2018concept)
* [Change Detection in Graph Streams by Learning Graph Embeddings on Constant-Curvature Manifolds](https://dzambon.github.com/publications/grattarola2019change)
* [Anomaly and Change Detection in Graph Streams through Constant-Curvature Manifold Embeddings](https://dzambon.github.com/publications/zambon2018anomaly)
* [Detecting Changes in Sequences of Attributed Graphs](https://dzambon.github.com/publications/zambon2017detecting)


## Tutorial

Please have a look at this notebook `tutorial.ipynb`.


## The package

The code is written in `python3`. 
In the package you will find following folders
* **`cdg/graph`** interface for datasets of graphs and dissimilarities.
* **`cdg/embedding`** several types of numeric representations of graphs, such as, dissimilarity representation and manifold embeddings. 
* **`cdg/changedetection`** tests for change detection.
* **`cdg/utils`** utilities for the module.
* **`cdg/simulation`** utilities for repeated experiments.



## Requirements and suggested packages

You may need:
* `networkx` (available `pip install networkx`)
* `GraKel` (available [here](https://github.com/ysig/GraKeL))
* `graph-matching-toolkit` (available [here](https://github.com/dzambon/graph-matching-toolkit)) to compute graph 
edit distances. A good place to save it is `./graph-matching-toolkit/graph-matching-toolkit.jar`.
* `graphviz` for visualize graphs.
* `ecp` R package (available [here](https://cran.r-project.org/web/packages/ecp/index.html)) in order to run certain multi change point methods.


## Installation

Go to your preferred directory, then 
```bash
git clone https://github.com/dzambon/cdg
cd cdg
sudo pip3 install -e .
```


## Credits

Author: [Daniele Zambon](https://dzambon.github.io)    
Affiliation: [Università della Svizzera italiana](https://inf.usi.ch)   
eMail: `daniele.zambon@usi.ch`   
Licence: _BSD-3-Clause_   


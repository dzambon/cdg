# Change Detection test on a stream of Graphs (CDG) #

Author: *Daniele Zambon*, 

Affiliation: *Università della Svizzera italiana* 

eMail: `daniele.zambon@usi.ch` 

Last Update: *05/10/2017* 


## Please cite 
Please consider citing our following paper [1,2].
```
@article{zambon2017concept,
  title={Concept Drift and Anomaly Detection in Graph Streams},
  author={Zambon, Daniele and Alippi, Cesare and Livi, Lorenzo},
  journal={arXiv preprint arXiv:1706.06941},
  year={2017}
}
@inproceedings{zambon2017detecting,
  title={Detecting Changes in Sequences of Attributed Graphs},
  author={Zambon, Daniele and Livi, Lorenzo and Alippi, Cesare},
  booktitle={IEEE Symposium Series on Computational Intelligence},
  year={2017}
  organization={IEEE}
}
```
 



## Package description ##

The package is the reference code for the publications [1] and [2]. As such it implements the method and the code for runnig the experiments therein described. 
Please consider paper [1] for citations.

The code is written in `python` language and has been runned on a `Ubuntu 16.04 LTS` machine with python `3.5.2`.
The package is composed of `cdg` module, some scripts (`./scritps/`) and demos (`./demo/`).

### `cdg` Module ###

The module is composed of four files:

* **`graph.py`** interface for graph instances and some functionalities for computing features and finding graph prototypes.
* **`changedetection.py`** change detection tests.
* **`database.py`** interface for different datasets. 
* **`simulation.py`** pipeline of the change detection tests on stream of graphs presented in [1] .

### Script ###

See the help

```python3 ./scripts/process_results.py -h```


### Demo ###

See the help

```python3 ./demo/demo.py -h```

## Requirements ##

### Python libraries ###
You will need the following python3 packages:

* system: `sys`, `os`, `subprocess`, `pickle`, `datetime`
* numerical: `random`, `numpy`, `scipy`, `random`
* graphical: `matplotlib`
* graph: `pydotplus`, `xml`


### Other toolkits ###

Other required packages are:

* graph edit distance: `GraphMatchingToolkit` [3] (available here [graph-matching-toolkit](https://github.com/dan-zam/graph-matching-toolkit)).
* graph visualisation: `graphviz` [4].



## References ##

[1] Zambon, Daniele, Cesare Alippi, and Lorenzo Livi. "Concept Drift and Anomaly Detection in Graph Streams." arXiv preprint arXiv:1706.06941 (2017). Submitted.

[2] Zambon, Daniele, Lorenzo Livi, and Cesare Alippi. "Detecting Changes in Sequences of Attributed Graphs." IEEE Symposium Series on Computational Intelligence. 2017.

[3] Riesen, Kaspar, Sandro Emmenegger, and Horst Bunke. "A novel software toolkit for graph edit distance computation." International Workshop on Graph-Based Representations in Pattern Recognition. 2013.

[4] http://www.graphviz.org 


## Licence ##

Licence: BSD-3-Clause

Copyright (c) 2017, Daniele Zambon
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Conversion between graph formats.
# --------------------------------------------------------------------------------
from cdg.graph import Graph as cdg_Graph

import networkx as nx
import grakel
import numpy as np
from tqdm import tqdm

GRAPH_FORMATS_IMPORT = ['cdg', 'npy', 'grakel', 'nx', 'gxl']
GRAPH_FORMATS_EXPORT = ['cdg', 'npy', 'grakel', 'nx']

def infer_format(graphs):
    '''
    Infers the format of the graphs.
    :param graphs:
    :return:
    '''
    if isinstance(graphs[0], cdg_Graph):
        f = 'cdg'
    elif isinstance(graphs[0], nx.Graph):
        f = 'nx'
    elif isinstance(graphs[0], grakel.Graph):
        f = 'grakel'
    elif isistance_npy(graphs):
        f = 'npy'
    else:
        f = None
        # raise ValueError('Format of graph list {} not recognised'.format(graphs))
    return f

def isistance_npy(graphs):
    if not isinstance(graphs, list):
        return False
    elif len(graphs)!=3:
        return False
    elif not isinstance(graphs[0], np.ndarray) and \
         not isinstance(graphs[1], np.ndarray) and \
         not isinstance(graphs[2], np.ndarray):
        return False
    elif graphs[0].ndim!=3 and graphs[1].ndim!=3 and graphs[2].ndim!=4:
        return False
    else:
        return True
    
def convert(graphs, format_in, format_out, directed=False, label_ndim=None, **kwargs):
    # if format_in == 'nx':
    #     return Graph.export_to(graphs=graphs, format=format_out, directed=directed, label_ndim=label_ndim)
    # else:
    #     return Graph.import_from(graphs=graphs, format=format_in, directed=directed, label_ndim=label_ndim)
    
    assert format_in in GRAPH_FORMATS_IMPORT, 'format_in={} is not in {}, '.format(format_in, GRAPH_FORMATS_IMPORT)
    assert format_out in GRAPH_FORMATS_EXPORT, 'format_out={} is not in {}, '.format(format_out, GRAPH_FORMATS_EXPORT)
    assert format_in == infer_format(graphs), 'provided graphs are not in format format_in.'

    if format_in == format_out:
        conv_fun = identity
    elif format_in == 'cdg' and format_out == 'nx':
        conv_fun = cdg_to_nx
    elif format_in == 'cdg' and format_out == 'grakel':
        conv_fun = cdg_to_grakel
    elif format_in == 'grakel' and format_out == 'cdg':
        conv_fun = grakel_to_cdg
    elif format_in == 'npy' and format_out == 'cdg':
        conv_fun = npy_to_cdg
    elif format_in == 'nx' and format_out == 'cdg':
        conv_fun = nx_to_cdg
    elif format_in == 'gxl' and format_out == 'cdg':
        conv_fun = gxl_to_cdg
    elif format_in == 'nx' and format_out == 'npy':
        conv_fun = nx_to_npy
    elif format_in == 'gxl' and format_out == 'npy':
        conv_fun = gxl_to_npy
    elif format_in == 'gxl' and format_out == 'nx':
        conv_fun = gxl_to_nx
    else:
        raise NotImplementedError
    
    if not isinstance(graphs, list):
        graphs = [graphs]
    return conv_fun(graphs, directed=directed, label_ndim=label_ndim, **kwargs)
   
def identity(graphs, directed, label_ndim, **kwargs):
    return graphs
    
def cdg_to_nx(graphs, directed, label_ndim, **kwargs):
    with_nf = False if graphs[0].nf is None or graphs[0].nf.shape[1] == 0 else True
    with_ef = False if graphs[0].ef is None or graphs[0].ef.shape[1] == 0 else True

    graphs_conv = []
    for g in graphs:
        if directed:
            graph_class = nx.DiGraph
        else:
            graph_class = nx.Graph
            assert np.all(g.adj == g.adj.T)
        g_new = graph_class(g.adj)
        if label_ndim == 0:
            g.nf = g.nf if g.nf.ndim == 1 else g.nf[..., 0]
            g.ef = g.ef if g.ef.ndim == 2 else g.ef[..., 0]
        if with_nf:
            for vi in range(len(g_new.nodes)):
                g_new.nodes[vi]['vec'] = g.nf[vi]
        if with_ef:
            edge_list = list(g_new.edges)
            for ei in range(len(edge_list)):
                g_new.edges[edge_list[ei]]['vec'] = g.ef[edge_list[ei]]
        graphs_conv.append(g_new)
    return graphs_conv

def cdg_to_grakel(graphs, directed, label_ndim, **kwargs):

    has_node_attr = False if graphs[0].nf is None or graphs[0].nf.shape[1] == 0 else True
    has_edge_attr = False if graphs[0].ef is None or graphs[0].ef.shape[2] == 0 else True

    g_grakel = []
    for g in graphs:

        wh = np.where(g.adj == 1)
        edge_set = set()
        node_set = set()
        for i in range(len(wh[0])):
            node_set.add(wh[0][i])
            node_set.add(wh[1][i])
            edge_set.add((wh[0][i], wh[1][i]))

        if has_node_attr:
            node_attr_dict = {}
            for n in node_set:
                node_attr_dict[n] = g.nf[n]
        else:
            node_attr_dict = None

        if has_edge_attr:
            edge_attr_dict = {}
            for e in edge_set:
                edge_attr_dict[e] = g.ef[e[0], e[1]]
        else:
            edge_attr_dict = None

        g_grakel.append(grakel.Graph(initialization_object=edge_set,
                                     node_labels=node_attr_dict,
                                     edge_labels=edge_attr_dict))

    return g_grakel
    # g_nx = npy_to_nx(graphs, directed=directed, label_ndim=label_ndim)
    # return list(grakel.graph_from_networkx(g_nx, node_labels_tag='vec', edge_labels_tag='vec', as_Graph=True))

def grakel_to_cdg(graphs, directed, label_ndim, **kwargs):

    graphs_conv = []
    for g in graphs:
        
        N = len(g.vertices)
        
        # map grakel vertex id to set {0, 1, ..., N-1}
        vertices_from_zeros = np.array(list(g.vertices)).astype(int)
        
        def v2id(v):
            if isinstance(v, tuple) and len(v) == 2:
                return (v2id(v[0]), v2id(v[1]))
            else:
                return np.where(vertices_from_zeros == v)[0][0]
        
        # Adjacency matrix
        A = g.get_adjacency_matrix()
        
        # Feature matrices from grakel labels
        def grakel_get_features(N, entities, labels, **kwargs):
            '''
            Takes a set of entities, read their lables and store in matrix/tensor form.
            '''
            F = None
            for e in entities:
                # read label
                lab = labels[e]
                # parse label (F_current is the dimension of the feature space)
                if isinstance(lab, list):
                    lab = np.array(lab)
                elif isinstance(lab, int) or isinstance(lab, float):
                    lab = np.array([lab])
                elif isinstance(lab, np.ndarray):
                    lab = lab.ravel()
                else:
                    raise ValueError('I couldn\'t parse label {}'.format(lab))
                F_current = lab.shape[0]
                
                # Define the feature mat at first iteration
                if F is None:
                    F = F_current
                    if isinstance(e, tuple):  # is edge
                        features = np.zeros((N, N, F))
                    else:  # is node
                        features = np.zeros((N, F))
                
                assert F == F_current
                
                # Store label
                features[v2id(e)] = lab
            return features
        
        # Parse node labels
        if g.node_labels is None:
            X = None
        else:
            X = grakel_get_features(N, g.vertices, g.node_labels)
        
        # Parse edge labels
        if g.edge_labels is None:
            E = None
        else:
            edges = [ed[0] for ed in g.get_edges(purpose='dictionary')]
            E = grakel_get_features(N, edges, g.edge_labels)
        
        graphs_conv.append(cdg_Graph(A, X, E))
    return graphs_conv

def gxl_to_nx(graphs, directed, label_ndim, **kwargs):
    verbose = kwargs.pop('verbose', False)
    
    import subprocess
    import tempfile
    import networkx.drawing.nx_pydot

    graphs_conv = []
    for g in tqdm(graphs, desc='conv gxl to nx', disable=not verbose):
        proc = subprocess.Popen(["gxl2dot", g], stdout=subprocess.PIPE)
        dot_byte = proc.stdout.read()
        # print(dot_byte.decode())
        dot_file_temp = tempfile.NamedTemporaryFile(prefix="dot_convert_")  # 2
        dot_file_temp.write(dot_byte)  # 3
        dot_file_temp.seek(0)
        g_nx = networkx.drawing.nx_pydot.read_dot(dot_file_temp.name)
        dot_file_temp.close()
        graphs_conv.append(g_nx)
    return graphs_conv


def nx_to_npy(graphs, directed, label_ndim, **kwargs):
    import spektral.utils
    
    nf_keys = kwargs.pop('nf_keys')
    ef_keys = kwargs.pop('ef_keys')
    nf_preprocessing = kwargs.pop('nf_preprocessing')
    ef_preprocessing = kwargs.pop('ef_preprocessing')
    nf_postprocessing = kwargs.pop('nf_postprocessing')
    ef_postprocessing = kwargs.pop('ef_postprocessing')
    

    for g_nx in graphs:
        for v in g_nx.nodes:
            for i in range(len(nf_keys)):
                g_nx.node[v][nf_keys[i]] = nf_preprocessing[i](g_nx.node[v][nf_keys[i]])
        
        for e in g_nx.edges:
            for i in range(len(nf_keys)):
                g_nx.get_edge_data(e[0], e[1])[0][ef_keys[i]] = ef_preprocessing[i] \
                    (g_nx.get_edge_data(e[0], e[1])[0][ef_keys[i]])
    
    A, X, E = spektral.utils.nx_to_numpy(graphs, nf_keys=nf_keys, ef_keys=ef_keys, auto_pad=True,
                                      nf_postprocessing=nf_postprocessing, ef_postprocessing=ef_postprocessing)
    return A, X, E

def npy_to_cdg(graphs, directed, label_ndim, **kwargs):
    A = graphs[0]
    X = graphs[1]
    E = graphs[2]
    return [cdg_Graph(A[i], X[i], E[i]) for i in range(A.shape[0])]

def nx_to_cdg(graphs, directed, label_ndim, **kwargs):
    graphs_npy = nx_to_npy(graphs=graphs, directed=directed, label_ndim=label_ndim, **kwargs)
    return npy_to_cdg(graphs=graphs_npy, directed=directed, label_ndim=label_ndim, **kwargs)

def gxl_to_npy(graphs, directed, label_ndim, **kwargs):
    g_nx = gxl_to_nx(graphs, directed=directed, label_ndim=label_ndim, **kwargs)
    return nx_to_npy(g_nx, directed=directed, label_ndim=label_ndim, **kwargs)

def gxl_to_cdg(graphs, directed, label_ndim, **kwargs):
    g_nx = gxl_to_nx(graphs, directed=directed, label_ndim=label_ndim, **kwargs)
    return nx_to_cdg(g_nx, directed=directed, label_ndim=label_ndim, **kwargs)



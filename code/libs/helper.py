__author__ = 'lisette.espin'

##########################################################################################
# Dependencies
##########################################################################################

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from fast_pagerank import pagerank

##########################################################################################
# Hello world!
##########################################################################################

def test():
    '''
    toy-function that simply prints out the message 'hello world'
    '''
    print("hello world")
    
    
##########################################################################################
# Numpy array helpers
##########################################################################################

def is_symmetric(a, tol=1e-8):
    '''
    Given a numpy array A, it returns True if the matrix is symmetric False otherwise.
    '''
    return np.all(np.abs(a-a.T) < tol)

def make_symmetric(A):
    '''
    Given a numpy array A, it returns its symmetric version via the maximum method.
    '''
    return np.maximum( A, A.transpose() )

def get_adjacency_from_pandas_weighted_edgelist(df, nodes_order=None, directed=False):
    '''
    Dataframe df must have 3 columns (regardless of column name)
    First column: source
    Second column: target
    Third column: weight
    '''
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_weighted_edges_from(df.values)
    nodes_order = G.nodes() if nodes_order is None else nodes_order
    return nx.adjacency_matrix(G=G, nodelist=nodes_order).toarray()


##########################################################################################
# MONADIC HYPOTHESES
# returns a matrix based on vector values.
##########################################################################################

###############################################
# Monadic hypotheses: similarity functions
###############################################

def compare_same(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    same: whether vals is equal to valt
    '''
    return int(vals==valt)

def compare_different(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    different: whether vals is NOT equal to valt
    '''
    return int(vals!=valt)

def compare_inverted_distance(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    inverted_distance: 1 / abs(vals-valt)
    The smaller the difference between vals and valt, 
    the higher the returned value.
    A small difference between vals and valt means that,
    the source and target have very similar attribute values vals and valt.
    '''
    return 1/(abs(vals-valt)+0.00001)

def compare_subs_target_source(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    subs_target_source: performs target-value - source-value
    '''
    return valt-vals

def compare_subs_source_target(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    subs_source_target: performs source-value - target-value
    '''
    return vals-valt

def compare_target_value(vals, valt):
    '''
    Handler for node-node comparison.
    Atribute value vals of source node gets compared 
    with the attribute value valt of target node.
    
    target_value: simply returns target-value
    '''
    return valt


###############################################
# Monadic hypotheses: matrix
###############################################

def get_monadic_hypothesis(df, keyid, attribute, symmetric=False, comparison_fnc=compare_same, keyorder=None, dtype=np.float):
    '''
    Given a dataframe df, it focuses only on two columns: keyid and attribute.
    First, an nxn matrix is created.
    Then, each node i (row/source) is compared with each node j (column/target)
    by applying the function comparison_fnc (whose inputs are nodes i and j attribute values).
    '''
    
    # we take the only 2 columns we need
    tmp = df[[keyid,attribute]].copy()
    
    # we make the main column index (for better and rapid access to attriubte value)
    tmp.set_index(keyid, inplace=True)
    
    # we create an empty numpy array of size nxn (the number of unique keyids)
    n = tmp.shape[0]
    A = np.zeros((n,n), dtype=dtype)
    
    # if the order of nodes is not given,
    # then we take the one from the dataframe.
    if keyorder is None:
        keyorder = tmp[keyid].unique()
        
    # we traverse nodes in 2 for-loops (source, target)
    for s,source in enumerate(keyorder):
        for t,target in enumerate(keyorder):
            # we only traverse half of the matrix (for fast computation)
            if t > s:
                A[s,t] = comparison_fnc(tmp.loc[source,attribute], tmp.loc[target,attribute])
                A[t,s] = A[s,t] if symmetric else comparison_fnc(tmp.loc[target,attribute], tmp.loc[source,attribute])
                    
    return A



##########################################################################################
# NODE-LEVEL HYPOTHESES
# returns a vector based on node properties (e.g., degree, density, cenrality, etc)
##########################################################################################

###############################################
# Node-level hypothesis: ego functions
###############################################

def ego_density_directed(adjacency):
    '''
    Given the adjacency matrix, it returns the density of the 1-hop neighborhood of each node.
    It assumes the network is directed.
    '''
    a = adjacency.copy()
    np.fill_diagonal(a, 0)
    hop2 = a.dot(a)
    np.fill_diagonal(hop2, 0)
    return hop2.sum(axis=1) / (a.shape[0]-1)**2
    
def ego_density_undirected(adjacency):
    '''
    Given the adjacency matrix, it returns the density of the 1-hop neighborhood of each node.
    It assumes the network is undirected.
    '''
    return ego_density_directed(adjacency) / 2

def ego_degree(adjacency):
    '''
    Given the adjacency matrix, it returns the degree of every node as a numpy array.
    '''
    return (adjacency>0).astype(int).sum(axis=1)

def ego_pagerank(adjacency):
    '''
    Given the adjacency matrix, it returns pagerank of every node as a numpy array.
    '''
    return pagerank(csr_matrix(adjacency))

###############################################
# Node-level hypothesis: vector
###############################################

def get_ego_hypothesis(adjacency, ego_fnc=ego_density_undirected, missing=0, dtype=np.float):
    '''
    Given the adjacency matrix of the network, 
    this function measures an ego property (a structural property of each node),
    and returns a vector, where each cell is the calculated metric for each node.
    The metric is determined by the function ego_fnc.
    '''
    # Computes metric for each node
    V = ego_fnc(adjacency)
    
    # missing values
    missing_val = missing(V[np.where(~np.isnan(V))]) if callable(missing) else missing
    V[np.where(np.isnan(V))] = missing_val
    
    return V

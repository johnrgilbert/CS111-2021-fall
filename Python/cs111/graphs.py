import numpy as np
import numpy.linalg as npla
import scipy
import scipy.linalg as spla
import networkx as nx

#############################################################################
# Generate the Laplacian matrix of a path                                   #
#############################################################################

def path(n):
    """Laplacian matrix of the n-vertex path graph"""
    E = np.diag(np.ones(n-1), -1)
    L = 2*np.eye(n) - E - E.T
    L[0,0] = 1
    L[-1,-1] = 1

    return L
# end of path()


#############################################################################
# Read graph edges and vertex coords as a networkx Graph                    #
#############################################################################

def read_mesh(filebase):
    """Read in the edges and vertex coordinates of a graph as a 2D mesh.
    
    This reads one edge per line from 'filebase.edges',
    where an edge is just two 0-based vertex numbers,
    and reads an x coordinate and y coordinate from 'filebase.xy'.
    
    It returns a networkx undirected graph and a dictionary of vertex coordinates.
    
    We are careful to construct the graph so that the names of the vertices are 
    sequential integers beginning with 0, so that the graph's vertex iterator will
    create things like the Laplacian matrix in the order we expect.
    """
    
    G = nx.Graph()
    n = 0 
    coords = {}
    for line in open(filebase + '.xy'):
        G.add_node(n)
        x = float(line.split()[0])
        y = float(line.split()[1])
        coords[n] = (x, y)
        n += 1
    
    for line in open(filebase + '.edges'):
        v = int(line.split()[0])
        w = int(line.split()[1])
        G.add_edge(v,w)
        
    return (G, coords)
# end of read_mesh()

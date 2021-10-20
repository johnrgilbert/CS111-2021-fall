# From in-class transcript from Lecture 2, January 9, 2020

import numpy as np
import numpy.linalg as npla
import scipy 


#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 2 dimensions #
#############################################################################

def make_A(k):
    """Create the matrix of the discrete Laplacian operator in two dimensions on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**2-by-k**2 matrix representing the finite difference approximation to Laplace's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []
    for x in range(k):
        for y in range(k):
                
            # what row of the matrix is grid point (x,y)?
            row = x + k*y

            # the diagonal element in this row
            col = row
            triples.append((row, col, 4.0))
            # connect to grid neighbors in x dimension
            if x > 0:
                col = row - 1
                triples.append((row, col, -1.0))
            if x < k - 1:
                col = row + 1
                triples.append((row, col, -1.0))
            # connect to grid neighbors in y dimension
            if y > 0:
                col = row - k
                triples.append((row, col, -1.0))
            if y < k - 1:
                col = row + k
                triples.append((row, col, -1.0))

    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k*k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape = (ndim, ndim))
    
    return A 
# end of make_A


#############################################################################
# Make the temperature matrix (discrete Laplacian operator) in 3 dimensions #
#############################################################################
def make_A_3D(k):
    """Create the matrix for the 3-dimensional temperature problem on a k-by-k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**3-by-k**3 matrix representing the finite difference approximation to Poisson's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []
    for x in range(k):
        for y in range(k):
            for z in range(k):
                # what row of the matrix is grid point (i,j)?
                row = z + y*k + x*k*k
                # the diagonal element in this row
                triples.append((row, row, 6.0))
                # connect to grid neighbors in x dimension
                if x > 0:
                    triples.append((row, row - k*k, -1.0))
                if x < k - 1:
                    triples.append((row, row + k*k, -1.0))
                # connect to grid neighbors in y dimension
                if y > 0:
                    triples.append((row, row - k, -1.0))
                if y < k - 1:
                    triples.append((row, row + k, -1.0))
                # connect to grid neighbors in z dimension
                if z > 0:
                    triples.append((row, row - 1, -1.0))
                if z < k - 1:
                    triples.append((row, row + 1, -1.0))

    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k*k*k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = scipy.sparse.csr_matrix((values, (rownum, colnum)), shape = (ndim, ndim))
    
    return A 
# end of make_A_3D



#############################################################################
# Make a 16-by-16 version of the temperature matrix for demos               #
#############################################################################

def make_A_small():
    """Make a small k=4 version of the temperature matrix, as a dense array"""
    A = make_A(4)
    return A.toarray()
# end of make_A_small



#############################################################################
# Make a right-hand side vector for the 2D Laplacian / temperature matrix   #
#############################################################################

def make_b(k, top = 32, bottom = 32, left = 32, right = 32):
    """Create the right-hand side for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
      top: list of k values for top boundary (optional, defaults to 32)
      bottom: list of k values for bottom boundary (optional, defaults to 32)
      left: list of k values for top boundary (optional, defaults to 32)
      right: list of k values for top boundary (optional, defaults to 32)
    Outputs:
      b: the k**2 element vector (as a numpy array) for the rhs of the Poisson equation with given boundary conditions
    """
    # Start with a vector of zeros
    ndim = k*k
    b = np.zeros(shape = ndim)
    
    # Fill in the four boundaries as appropriate
    b[0        : k       ] += top
    b[ndim - k : ndim    ] += bottom
    b[0        : ndim : k] += left
    b[k-1      : ndim : k] += right
    
    return b
# end of make_b



#############################################################################
# Make a size-16 version of the right-hand side for demos                   #
#############################################################################

def make_b_small():
    """Make a small k=4 version of the right-hand side vector"""
    return make_b(4, top=radiator(4))
# End of make_b_small


    
#############################################################################
# Make one wall with a radiator                                             #
#############################################################################

def radiator(k, width = .3, rad_temp = 212., wall_temp = 32.):
    """Create one wall with a radiator
    Parameters: 
      k: number of grid points in each dimension; length of the wall
      width: width of the radiator as a fraction of length of the wall 
      rad_temp:  temperature of the radiator (default 212)
      wall_temp: temperature of the wall outside the radiator (default 32)
    Outputs:
      wall: the k element vector (as a numpy array) for the boundary conditions at the wall
    """
    rad_start = int(k * (0.5 - width/2))
    rad_end = int(k * (0.5 + width/2))
    wall = wall_temp * np.ones(k)
    wall[rad_start : rad_end] = rad_temp
    
    return wall
# End of radiator

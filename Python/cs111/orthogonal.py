import numpy as np
import numpy.linalg as npla
import scipy

#############################################################################
# Generate a random orthogonal matrix                                       #
#############################################################################

def random_orthog(nrows, ncols = None):
    """Generate a random n-by-n orthogonal matrix, or a random matrix with orthonormal columns
    Parameters: 
      nrows: number of rows
      ncols: number of columns (defaults to nrows, i.e. a square orthogonal matrix)
    Output:
      Q: the matrix
    """
    if ncols is None:
        ncols = nrows
    assert ncols <= nrows, "orthonormal matrix cannot have more columns than rows"

    A = np.random.randn(nrows, ncols)
    Q, R = scipy.linalg.qr(A, mode='economic')

    return Q
# end of random_orthog


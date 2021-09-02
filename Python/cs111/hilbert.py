import numpy as np

def hilbert(n):
    """n-by-n Hilbert matrix, a famous example of an ill-conditioned matrix"""
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 1 / (i + j + 1)
    return A
# end of hilbert

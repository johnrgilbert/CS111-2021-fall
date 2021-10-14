import numpy as np
import numpy.linalg as npla
import scipy

#############################################################################
# Solve Ax = b by Jacobi iterative method                                   #
#############################################################################

def Jsolve(A, b, tol = 1e-8, max_iters = 1000, callback = None):
    """Solve a linear system Ax = b for x by the Jacobi iterative method.
    Parameters: 
      A: the matrix.
      b: the right-hand side vector.
      tol: the relative residual at which to stop iterating.
      max_iters: the maximum number of iterations to do. 
      callback: a user function to call at every iteration. 
        The callback function has arguments 'x', 'iteration', and 'residual'
    Outputs (in order):
      x: the computed solution
      rel_res: list of relative residual norms at each iteration.
        The number of iterations actually done is len(rel_res) - 1
    """
    # Check the input
    m, n = A.shape
    assert m == n, "matrix must be square"
    bn, = b.shape
    assert bn == n, "rhs vector must be same size as matrix"

    # Make the matrix use the sparse csr data structure
    A = scipy.sparse.csr_matrix(A)

    # Split A into diagonal D plus off-diagonal C
    d = A.diagonal()         # diagonal elements of A as a vector
    # D = np.diag(d)           # diagonal of A as a matrix -- DON'T DO THIS, IT CREATES A HUGE DENSE ARRAY.
    C = A.copy()
    C.setdiag(np.zeros(n))   # A without the diagonal
    
    # Initial guess: x = 0
    x = np.zeros(n)

    # Vector of relative residuals
    # Relative residual is norm(residual)/norm(b)
    # Intitial residual is b - Ax for x=0, or b
    rel_res = [1.0]
        
    # Call user function if specified
    if callback is not None:
        callback(x = x, iteration = 0, residual = 1)

    # Iterate
    for k in range(1, max_iters+1):
        # New x
        x = (b - C @ x) / d

        # Record relative residual
        this_rel_res = npla.norm(b - A @ x) / npla.norm(b)
        rel_res.append(this_rel_res)
                
        # Call user function if specified
        if callback is not None:
            callback(x = x, iteration = k, residual = this_rel_res)
                        
        # Stop if within tolerance    
        if this_rel_res <= tol:
            break
            
    return (x, rel_res)
# end of Jsolve


#############################################################################
# Solve Ax = b by conjugate gradient method (for SPD A only)                #
#############################################################################

def CGsolve(A, b, tol = 1e-8, max_iters = 1000, callback = None):
    """Solve a linear system Ax = b for x by the conjugate gradient iterative method.
    Parameters: 
      A: the matrix.
      b: the right-hand side vector.
      tol: the relative residual at which to stop iterating.
      max_iters: the maximum number of iterations to do. 
      callback: a user function to call at every iteration.
        The callback function has arguments 'x', 'iteration', and 'residual'
    Outputs (in order):
      x: the computed solution
      rel_res: list of relative residual norms at each iteration.
        The number of iterations actually done is len(rel_res) - 1
    """
    # Check the input
    m, n = A.shape
    assert m == n, "matrix must be square"
    bn, = b.shape
    assert bn == n, "rhs vector must be same size as matrix"

    # Make the matrix use the sparse csr data structure
    A = scipy.sparse.csr_matrix(A)

    # Initial guess: x = 0
    x = np.zeros(n)
    
    # Initial residual: r = b - A@0 = b
    r = b
 
    # Initial step is in direction of residual.
    d = r

    # Squared norm of residual
    rtr = r.T @ r
    
    # Vector of relative residuals
    # Relative residual is norm(residual)/norm(b)
    # Intitial residual is b - Ax for x=0, or b
    rel_res = [1.0]
     
    # Call user function if specified
    if callback is not None:
        callback(x = x, iteration = 0, residual = 1)

    # Iterate
    for k in range(1, max_iters+1):
        Ad = A @ d
        alpha = rtr / (d.T @ Ad)  # Length of step
        x = x + alpha * d         # Update x to new x
        r = r - alpha * Ad        # Update r to new residual
        rtrold = rtr 
        rtr = r.T @ r
        beta = rtr / rtrold    
        d = r + beta * d          # Update d to new step direction
                   
        # Record relative residual
        this_rel_res = npla.norm(b - A @ x) / npla.norm(b)
        rel_res.append(this_rel_res)
                
        # Call user function if specified
        if callback is not None:
            callback(x = x, iteration = k, residual = this_rel_res)
                        
        # Stop if within tolerance    
        if this_rel_res <= tol:
            break
            
    return (x, rel_res)
#end of CGsolve

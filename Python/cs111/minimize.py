import autograd.numpy as np
import autograd.numpy.linalg as npla
from autograd import grad
import scipy

#############################################################################
# Minimize a function of n variables by gradient descent.                   #
#############################################################################

def gradient_descent(func, x0, history=True, tol=1e-4, rate=.01, max_iters=1000, callback=None, **kwargs):
    """CS 111 toy gradient descent implementation
    
    Inputs:
        func:      callable objective function, takes a point x (as an ndarray) 
                   and returns the pair (value, gradient vector) 
        x0:        starting point, numpy array
        history:   if True, return list of xks and costs at each iteration
        tol:       stop when xk changes by less than this
        rate:      learning rate
        max_iters: maximum number of steps to take
        callback:  callable, called with xk at each step, optional
        
    Output is a scipy.optimize.OptimizeResult object with fields:
        x:         solution of the optimization
        success:   bool, whether it thinks it worked
        message:   str, cause of termination
        nit:       int, number of iterations
        xks:       list of points at each iteration (if history is True)
        costs:     list of function values at each iteration (if history is True)
        
        """

    xk = x0
    
    # Set default result
    result = scipy.optimize.OptimizeResult()
    result['success'] = False
    result['message'] = "maximum iterations reached without convergence."
          
    # Call user function if specified
    if callback is not None:
        callback(xk)
        
    # Initialize history if specified
    if history:
        xks = [np.copy(xk)]
        costs = [func(x0)[0]]

    # Iterate
    for k in range(1, max_iters+1):
        fvalue, gradient = func(xk)
        delta_x = rate * (-gradient)
        xk = xk + delta_x
                
        # Call user function if specified
        if callback is not None:
            callback(xk)
                           
        # Record this point in history
        if history:
            xks.append(np.copy(xk))
            costs.append(fvalue)
            
        # Stopping condition    
        if npla.norm(delta_x) <= tol:
            result['success'] = True
            result['message'] = "optimization terminated successfully"
            break
            
    result['x']   = xk
    result['nit'] = k 
    if history:
        result['costs'] = costs
        result['xks']   = xks 
            
    return result

# end of gradient_descent


#############################################################################
# Minimize a function gradient descent accelerated with momentum.           #
#############################################################################

def gradient_momentum(func, x0, history=True, tol=1e-4, rate=.01, beta=0, max_iters=1000, callback=None, **kwargs):
    """CS 111 toy gradient descent with momentum
    
    Inputs:
        func:      callable objective function, takes a point x (as an ndarray) 
                   and returns the pair (value, gradient vector) 
        x0:        starting point, numpy array
        history:   if True, return list of xks and costs at each iteration
        tol:       stop when xk changes by less than this
        rate:      learning rate
        max_iters: maximum number of steps to take
        callback:  callable, called with xk at each step, optional
        
    Output is a scipy.optimize.OptimizeResult object with fields:
        x:         solution of the optimization
        success:   bool, whether it thinks it worked
        message:   str, cause of termination
        nit:       int, number of iterations
        xks:       list of points at each iteration (if history is True)
        costs:     list of function values at each iteration (if history is True)
        
        """

    xk = x0
    
    # Set default result
    result = scipy.optimize.OptimizeResult()
    result['success'] = False
    result['message'] = "maximum iterations reached without convergence."
          
    # Call user function if specified
    if callback is not None:
        callback(xk)
        
    # Initialize history if specified
    if history:
        xks = [np.copy(xk)]
        costs = [func(x0)[0]]

    # Iterate
    previous_direction = 0
    for k in range(1, max_iters+1):
        fvalue, gradient = func(xk)
        direction = - gradient + beta * previous_direction
        previous_direction = direction
        delta_x = rate * direction
        xk = xk + delta_x
                
        # Call user function if specified
        if callback is not None:
            callback(xk)
                           
        # Record this point in history
        if history:
            xks.append(np.copy(xk))
            costs.append(fvalue)
            
        # Stopping condition    
        if npla.norm(delta_x) <= tol:
            result['success'] = True
            result['message'] = "optimization terminated successfully"
            break
            
    result['x']   = xk
    result['nit'] = k 
    if history:
        result['costs'] = costs
        result['xks']   = xks 
            
    return result

# end of gradient_momentum

# These are the classroom demo versions of Euler's method
# and the 2-slope method. They are not robust enough for
# general use (not even close), they're just demos.

import numpy as np
import scipy
from scipy import integrate

# ODE1 : Simple forward Euler method, fixed step size
def ode1(fun, t_span, y0, h):
    """Forward Euler algorithm: demo version
    ode1(fun, t_span, y0, h) uses fixed step size h
    This is only demo code, don't use it for real!
    """
    
    # First make the inputs into numpy arrays
    t0     = np.array(t_span[0]).reshape(1)
    tfinal = np.array(t_span[1]).reshape(1)
    y0     = np.array(y0).reshape(len(y0), 1)
    
    # Initialize the list of solution points
    sol_t = t0
    sol_y = y0

    step = 0
    t = t0
    y = y0
    while t < tfinal:
        s1 = np.array(fun(t, y))
        y = y + h * s1
        t = t + h
        sol_t = np.concatenate((sol_t, t))
        sol_y = np.concatenate((sol_y, y), axis = 1)
        step += 1
    print('ode1 took', step, 'steps')
    return sol_t, sol_y
# end of ode1

def ode2(fun, t_span, y0, h):
    """Modified Euler algorithm that uses two slopes
    ode2(fun, t_span, y0, h) uses fixed step size h
    This is only demo code, don't use it for real!
    """
    
    # First make the inputs into numpy arrays
    t0     = np.array(t_span[0]).reshape(1)
    tfinal = np.array(t_span[1]).reshape(1)
    y0     = np.array(y0).reshape(len(y0), 1)
    
    # Initialize the list of solution points
    sol_t = t0
    sol_y = y0

    step = 0
    t = t0
    y = y0
    while t < tfinal:
        s1 = np.array(fun(t, y))
        s2 = np.array(fun( t + (h/2), y + (h/2)*s1))
        y = y + h * s2
        t = t + h
        
        sol_t = np.concatenate((sol_t, t))
        sol_y = np.concatenate((sol_y, y), axis = 1)
        step += 1
    print('ode2 took', step, 'steps')
    return sol_t, sol_y
# end of ode2


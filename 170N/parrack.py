import numpy as np

def dynamics_solve(f, Dim = 1, t_0 = 0.0, s_0 = 1, dt = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        dt: Time step (float) set to 0.1 as default
        N: Number of time steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    
    T = np.array([t_0 + n * dt for n in range(N + 1)])
    
    if Dim == 1:
        S = np.zeros(N + 1)
    
    if Dim > 1:
        S = np.zeros((N + 1, Dim))
        
    S[0] = s_0
    
    if method == 'Euler':
        for n in range(N):
            S[n + 1] = S[n] + dt * f(T[n], S[n])
    
    if method == 'RK2':
        for n in range(N):
            k1 = dt * f(T[n], S[n])
            k2 = dt * f(T[n] + dt/2, S[n] + k1/2)
            S[n + 1] = S[n] + k2
    
    if method == 'RK4':
        for n in range(N):
            k1 = f(T[n], S[n])
            k2 = f(T[n] + dt/2, S[n] + (dt/2 * k1))
            k3 = f(T[n] + dt/2, S[n] + (dt/2 * k2))
            k4 = f(T[n] + dt, S[n] + (dt * k3))
            S[n + 1] = S[n] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
    return T, S


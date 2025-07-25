import numpy as np
import numba as nb

@nb.njit
def stochRK4(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):

    n = len(y_0)
    assert len(fn(t_0, y_0, parameters)) == n, "Field function return mismatch w.r.t. y_0!"
    
    lh = h / 2. # Define leapfrog "half step"

    ts       = np.arange(t_0, t_end, lh)
    ys       = np.zeros(shape = (len(ts), n), dtype = np.float64)
    ys[0,:]  = y_0
    
    stoch_step = False
    
    for i, t in enumerate(ts[:-1]):
        if stoch_step:
            noise = np.array([np.random.normal() if noise_mask[i] else 0. for i in range(n)],
                                 dtype = np.float64)
            ys[i,:] += np.sqrt(2 * T * h) * noise # Balanced scheme of step h.
        k1       = fn(t           , ys[i,:]                , parameters)
        k2       = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k1, parameters) 
        k3       = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k2, parameters) 
        k4       = fn(t + lh      , ys[i,:] +       lh * k3, parameters) 
        ys[i+1,:] = ys[i,:] + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        stoch_step = not stoch_step
    return ts, ys

@nb.njit
def stochRK4_nonnegative(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):
    # Modified algorithm that constrains the coordinates in the nonnegative hyperquadrant.
    n = len(y_0)
    assert len(fn(t_0, y_0, parameters)) == n, "Field function return mismatch w.r.t. y_0!"
    
    lh = h / 2. # Define leapfrog "half step"

    ts       = np.arange(t_0, t_end, lh)
    ys       = np.zeros(shape = (len(ts), n), dtype = np.float64)
    ys[0,:]  = y_0
    assert np.all(ys[0,:] >= 0.), "y_0 inconsistent with nonnegativity of trajectory!"  

    stoch_step = False
    
    for i, t in enumerate(ts[:-1]):
        if stoch_step:
            noise = np.array([np.random.normal() if noise_mask[i] else 0. for i in range(n)],
                                 dtype = np.float64)
            ys[i,:] += np.sqrt(2 * T * h) * noise # Balanced scheme of step h.
            ys[i,:][ys[i,:] < 0.] = 0. # Constraint implementation 

        k1       = fn(t           , ys[i,:]                , parameters)
        k2       = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k1, parameters) 
        k3       = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k2, parameters) 
        k4       = fn(t + lh      , ys[i,:] +       lh * k3, parameters) 
        ys[i+1,:] = ys[i,:] + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        
        ys[i+1,:][ys[i+1,:] < 0.] = 0. # Constraint implementation 

        stoch_step = not stoch_step
    return ts, ys

@nb.njit
def stochRK4_event_detection(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):
    # Modified algorithm containing also a event switch boolean.
    # Integration terminates whenever the event boolean is triggered.
    # Expects an additional boolean after the vector field returned by fn, which
    # if True is considered as an integration-terminating event.
    event = False
    n  = len(y_0)
    assert len(fn(t_0, y_0, parameters)[0]) == n
    
    lh = h / 2. # Define leapfrog "half step"

    ts       = np.arange(t_0, t_end, lh)
    ys       = np.zeros(shape = (len(ts), len(y_0)), dtype = np.float64)
    ys[0,:]  = y_0
    
    stoch_step = False
    
    for i, t in enumerate(ts[:-1]):
        if stoch_step:
            noise = np.array([np.random.normal() if noise_mask[i] else 0. for i in range(n)],
                                 dtype = np.float64)
            ys[i,:] += np.sqrt(2 * T * h) * noise # Balanced scheme of step h. 
        k1, event = fn(t           , ys[i,:]                , parameters)
        k2, _     = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k1, parameters) 
        k3, _     = fn(t + 0.5 * lh, ys[i,:] + 0.5 * lh * k2, parameters) 
        k4, _     = fn(t + lh      , ys[i,:] +       lh * k3, parameters) 
        ys[i+1,:] = ys[i,:] + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        stoch_step = not stoch_step
        if event == True:
            t_ret = t
            y_ret = ys[i]
            return t, ys[i], event
    t_ret = ts[-1]
    y_ret = ys[-1]
    return t_ret, y_ret, False

@nb.njit
def stochRK4_event_times(fn, event_location, n_events, interevent_cooldown, h, y_0, T, 
                         noise_mask, parameters, t_0 = 0., verbose = False):
    # Runge-Kutta 4 with event detection that does not return the system
    # trajectory, nor keeps track of it except for its current state.
    # Meant for simulation of extremely low-rate events.
    # Stops when a given amount of events is recorded.
    # The meaning of the event is different from that of stochRK4, as it is
    # specified by the integrator, and no return of a boolean is expected by the function.
    event = False
    n  = len(y_0)
    assert len(fn(t_0, y_0, parameters)) == n
    
    lh = h / 2. # Define leapfrog "half step"

    t        = t_0
    y        = y_0
    next_y   = 0. 

    stoch_step = False
    
    event_times = []

    if T <= 1e-4:
        print("WARNING: small T, there may be no events!")
    while True:
        if stoch_step:
            noise = np.array([np.random.normal() if noise_mask[i] else 0. for i in range(n)],
                                 dtype = np.float64)
            y += np.sqrt(2 * T * h) * noise # Balanced scheme of step h. 
        k1 = fn(t           , y                , parameters)
        k2 = fn(t + 0.5 * lh, y + 0.5 * lh * k1, parameters) 
        k3 = fn(t + 0.5 * lh, y + 0.5 * lh * k2, parameters) 
        k4 = fn(t + lh      , y +       lh * k3, parameters) 
        next_y = y + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        stoch_step = not stoch_step

        event = (next_y[0] > event_location) and (y[0] < event_location) 
        if event == True:
            if verbose:
                print("New event @",t," logged as #", len(event_times)+1)
            if not len(event_times):
                event_times.append(t)
            elif (t - event_times[-1]) > interevent_cooldown:
                event_times.append(t)
            elif verbose:
                print("Event was rejected as too close to previous event.")
        t += lh
        y = next_y
        if len(event_times) >= n_events:
            return np.array(event_times)

@nb.njit
def euler_maruyama(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):
    n  = len(y_0)
    assert len(fn(t_0, y_0, parameters)) == n

    ts       = np.arange(t_0, t_end, h)
    ys       = np.empty(shape = (len(ts), len(y_0)), dtype = np.float64)
    ys[0,:]  = y_0
    
    noise    = np.empty(shape = (len(ts), len(y_0)), dtype = np.float64)
    for i in range(n):
        if noise_mask[i]:
            noise[:,i] = np.random.normal(loc = 0., scale = np.sqrt(2 * T * h), size = len(ts))
        else:
            noise[:,i] = np.zeros(shape = (len(ts)))
    
    for i, t in enumerate(ts[:-1]):
        ys[i+1,:] = ys[i,:] + h * fn(t, ys[i,:], parameters) + noise[i,:]
    return ts, ys

@nb.njit
def euler_maruyama_event_detection(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):
    # Modified algorithm containing also a event switch boolean.
    # Integration terminates whenever the event boolean is triggered.
    # Expects an additional boolean after the vector field returned by fn, which
    # if True is considered as an integration-terminating event.
    event = False
    n  = len(y_0)
    assert len(fn(t_0, y_0, parameters)[0]) == n

    ts       = np.arange(t_0, t_end, h)
    ys       = np.zeros(shape = (len(ts), len(y_0)), dtype = np.float64)
    ys[0,:]  = y_0
    
    noise    = np.empty(shape = (len(ts), len(y_0)), dtype = np.float64)
    for i in range(n):
        if noise_mask[i]:
            noise[:,i] = np.random.normal(loc = 0., scale = np.sqrt(2 * T * h), size = len(ts))
        else:
            noise[:,i] = np.zeros(shape = (len(ts)))

    for i, t in enumerate(ts[:-1]):
        field, event = fn(t, ys[i,:], parameters)
        ys[i+1,:] = ys[i,:] + h * field + noise[i,:]
        if event == True:
            t_ret = t
            y_ret = ys[i]
            return t_ret, y_ret, event
    t_ret = ts[-1]
    y_ret = ys[-1]
    return t_ret, y_ret, False

def ho(t, y, parameters):
    p, q = y
    omega = parameters[0]
    beta  = parameters[1]
    return np.array([omega * q - beta * p, - omega * p - beta * q])

@nb.njit
def fhn(t, y, parameters):
    u, v = y
    eps, a = parameters
    return np.array([(u - u**3 /3 - v)/eps, u + a])

import matplotlib.pyplot as plt

if __name__ == "__main__":
    sol = euler_maruyama(fhn, 100, 1e-3, 
                   np.array([1, 0]), 0.1/np.sqrt(1e-2), 
                   np.array([True, False]), np.array([1e-2, 1.3]))
    print(sol)
    plt.figure()
    plt.plot(sol[0], sol[1][:, 0], label = r"$p$")
    plt.plot(sol[0], sol[1][:, 1], label = r"$q$")
    plt.legend()
    plt.figure()
    plt.plot(sol[1][:, 0],sol[1][:, 1])
    plt.show()

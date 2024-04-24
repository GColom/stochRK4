import numpy as np

def stochRK4(fn, t_end, h, y_0, T, noise_mask, parameters, t_0 = 0.):

    N  = len(y_0)
    assert len(fn(t_0, y_0, parameters)) == N
    
    lh = h / 2. # Define leapfrog "half step"

    ts       = np.arange(t_0, t_end, lh)
    ys       = np.zeros(shape = (len(ts), len(y_0)), dtype = float)
    ys[0,:]  = y_0
    
    stoch_step = False
    
    for i, t in enumerate(ts):
        if stoch_step:
            noise = np.fromiter([np.random.normal() if noise_mask[i] else 0. for i in range(N)],
                                 dtype = float)
            ys[i,:] += np.sqrt(2 * T * h) * noise # Balanced scheme of step h. 
        k1       = fn(t        , ys[i,:]               , parameters)
        k2       = fn(t + lh/2., ys[i,:] + lh * k1 / 2., parameters) 
        k3       = fn(t + lh/2., ys[i,:] + lh * k2 / 2., parameters) 
        k4       = fn(t + lh   , ys[i,:] + lh * k3     , parameters) 
        try:
            ys[i+1,:] = ys[i,:] + lh * (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        except IndexError:
            return ts, ys
        stoch_step = not stoch_step
    return t, ys

def ho(t, y, parameters):
    p, q = y
    omega = parameters[0]
    beta  = parameters[1]
    return np.array([omega * q - beta * p, - omega * p - beta * q])

def fhn(t, y, parameters):
    u, v = y
    eps, a = parameters
    return np.array([(u - u**3 /3 - v)/eps, u + a])

import matplotlib.pyplot as plt

if __name__ == "__main__":
    sol = stochRK4(fhn, 100, 1e-3, np.array([1, 0]), 0.1/np.sqrt(1e-2), [True, False], [1e-2, 1.3])
    print(sol)
    plt.figure()
    plt.plot(sol[0], sol[1][:, 0], label = r"$p$")
    plt.plot(sol[0], sol[1][:, 1], label = r"$q$")
    plt.legend()
    plt.figure()
    plt.plot(sol[1][:, 0],sol[1][:, 1])
    plt.show()

# price_model.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_paths(double p0, double mu, np.ndarray[np.float64_t, ndim=1] sigma, double T, int N, int n_paths):
    cdef double dt = T / N
    cdef np.ndarray[np.float64_t, ndim=2] paths = np.zeros((n_paths, N), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] t = np.linspace(0, T, N)
    cdef np.ndarray[np.float64_t, ndim=2] dW
    cdef int i, j
    
    paths[:, 0] = p0
    dW = np.random.normal(0, sqrt(dt), (n_paths, N - 1))
    
    for i in range(n_paths):
        for j in range(1, N):
            paths[i, j] = paths[i, j-1] + mu * dt + sigma[j-1] * dW[i, j-1]
    
    return paths, t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_density(np.ndarray[np.float64_t, ndim=2] paths, int n_bins):
    cdef np.ndarray[np.float64_t, ndim=1] flat = paths.ravel()
    cdef np.ndarray[np.float64_t, ndim=1] hist
    cdef np.ndarray[np.float64_t, ndim=1] bins
    hist, bins = np.histogram(flat, bins=n_bins, density=True)
    return hist, bins

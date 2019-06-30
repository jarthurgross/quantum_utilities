import numpy as np

def dag(op):
    return np.conjugate(op.T)

import numpy as np

def dag(op):
    return np.conjugate(op.T)

def outer_prod(ketL, ketR):
    return np.outer(ketL, ketR.conj())

def inner_prod(ketL, ketR):
    return np.dot(ketL.conj(), ketR)

def rho_from_ket(ket):
    return outer_prod(ket, ket)

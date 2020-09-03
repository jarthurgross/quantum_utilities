import numpy as np
from scipy.special import factorial, eval_genlaguerre

def make_a(N):
    '''Make a truncated annihilation operator.

    Parameters
    ----------
    N: positive integer
        The dimension of the truncated Hilbert space, basis {0, ..., N-1}

    Returns
    -------
    numpy.array
        Annihilation operator in the truncated Hilbert space, represented in the
        number basis

    '''
    return np.diag(np.sqrt(np.arange(1, N, dtype=np.complex)), 1)

def get_displace_op_matrix_element(m, n, alpha):
    r'''Get matrix elements from a displacement operator.
    
    The number-basis elements are
    :math:`\langle m|\exp(\alpha a^\dagger - \alpha^* a)|n\rangle`

    Parameters
    ----------
    m: nonnegative integer
        Number index for the row of the matrix element
    n: nonnegative integer
        Number index for the column of the matrix element
    alpha: complex number
        Displacement amplitude

    Returns
    -------
    complex number
        Number-basis matrix element of the displacement operator

    '''
    abs_alpha_sq = np.abs(alpha)**2
    if m >= n:
        return (np.sqrt(factorial(n)/ factorial(m)) *
                np.exp(-abs_alpha_sq / 2) * alpha**(m - n) *
                eval_genlaguerre(n, m - n, abs_alpha_sq))
    return (np.sqrt(factorial(m)/ factorial(n)) *
            np.exp(-abs_alpha_sq / 2) * np.conj(-alpha)**(n - m) *
            eval_genlaguerre(m, n - m, abs_alpha_sq))

def make_displacement_op(alpha, N):
    r'''Make a truncated displacement operator.
    
    The displacement operator is :math:`\exp(\alpha a^\dagger - \alpha^* a)`

    Parameters
    ----------
    N: positive integer
        The dimension of the truncated Hilbert space, basis {0, ..., N-1}
    alpha: complex number
        Displacement amplitude

    Returns
    -------
    numpy.array
        Displacement operator in the truncated Hilbert space, represented in the
        number basis

    '''
    op = np.empty((N, N), dtype=np.complex)
    for j in range(N):
        for k in range(N):
            op[j,k] = get_displace_op_matrix_element(j, k, alpha)
    return op

def make_vac_state_vec(N):
    r'''Make a truncated vacuum-state vector.

    Parameters
    ----------
    N: positive integer
        The dimension of the truncated Hilbert space, basis {0, ..., N-1}

    Returns
    -------
    numpy.array
        Vacuum-state vector in the truncated Hilbert space, represented in the
        number basis

    '''
    ket = np.zeros(N, dtype=np.complex)
    ket[0] = 1
    return ket

def make_coh_state_vec(alpha, N, normalized=True):
    r'''Make a truncated coherent-state vector.

    The coherent-state vector is :math:`D(\alpha)|0\rangle`. The truncated
    vector is renormalized by default.

    Parameters
    ----------
    N: positive integer
        The dimension of the truncated Hilbert space, basis {0, ..., N-1}
    alpha: complex number
        Coherent-state amplitude
    normalized: boolean
        Whether or not the truncated vector is renormalized

    Returns
    -------
    numpy.array
        Coherent-state vector in the truncated Hilbert space, represented in the
        number basis

    '''
    ns = np.arange(0, N, dtype=np.complex)
    coh_vec = np.power(alpha, ns) / np.sqrt(factorial(ns))
    return coh_vec / np.linalg.norm(coh_vec) if normalized else coh_vec

def make_squeezed_state_vec(r, phi, N, normalized=True):
    r'''Make a truncated squeezed-state vector.

    The squeezed-state vector is :math:`S(r,\phi)|0\rangle`. The truncated
    vector is renormalized by default.

    Parameters
    ----------
    N: positive integer
        The dimension of the truncated Hilbert space, basis {0, ..., N-1}
    r: real number
        Squeezing amplitude
    phi: real number
        Squeezing phase
    normalized: boolean
        Whether or not the truncated vector is renormalized

    Returns
    -------
    numpy.array
        Squeezed-state vector in the truncated Hilbert space, represented in the
        number basis

    '''
    ket = np.zeros(N, dtype=np.complex)
    for n in range(N//2):
        ket[2*n] = (1 / np.sqrt(np.cosh(r))) * ((-0.5 * np.exp(2.j * phi) * np.tanh(r))**n /
                                                factorial(n)) * np.sqrt(factorial(2 * n))
    return ket / np.linalg.norm(ket) if normalized else ket

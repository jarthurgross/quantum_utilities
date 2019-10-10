import itertools as it
import numpy as np

def op_vec_to_matrix(op_vec, basis):
    '''Turn a vectorized operator back into a matrix.

    Parameters
    ----------
    op_vec: numpy.array
        Vector representing the operator in some basis
    basis: list(numpy.array)
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The matrix representation of the operator

    '''
    try:
        op_matrix = np.einsum('j,jmn->mn', op_vec, basis)
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        op_matrix = np.einsum('j,jmn->mn', op_vec, basis)
    return op_matrix

def proc_mat_to_LR_tensor(A, basis):# IP_jk = (X_j|X_k)
    '''Turn a process matrix into the corresponding left-right-action tensor.

    A process matrix acts on vectorized operators via matrix-vector
    multiplication: b_j*X_j -> A_jk*b_k*X_j.  A left-right-action tensor acts on
    operators by multiplying the basis operator corresponding to the left index
    by the Hilbert-Schmidt inner product of the operator argument with the basis
    operator corresponding to the right index: B -> C_jk*tr(X^dag_k*B)*X_j.

    Parameters
    ----------
    A: numpy.array
        The process matrix with respect to the given basis
    basis: list(numpy.array)
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The tensor whose left-right action is equivalent to the matrix-vector
        action of the process matrix in the given basis

    '''
    try:
        IP = np.einsum('jmn,kmn->jk', basis, np.conj(basis))
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        IP = np.einsum('jmn,kmn->jk', basis, np.conj(basis))

    # rho = r_j*X_j
    # E(rho) = A_jk*r_k*X_j
    # (X_j|E(X_k)) = A_jk*(X_j|X_j)
    # A_jk = (X_j|E(X_k))/(X_j|X_j)
    # E(rho) = E_jk*(X_k|rho)*X_j
    #        = E_jk*r_m*(X_k|X_m)*X_j
    #        = E_jm*(X_m|X_k)*r_k*X_j
    # E_jm*IP_mk = A_jk
    # E_jk = A_jm*IP.inv()_mk
    IPinv = np.linalg.inv(IP)
    E = A @ IPinv
    return E

def get_supdup_op(basis):
    # Superoperator change of basis
    # B_mnpq_jk = X_j_mn*X.conj()_k_qp
    try:
        B = np.einsum('jmn,kqp->mnpqjk', basis, np.conj(basis))
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        B = np.einsum('jmn,kqp->mnpqjk', basis, np.conj(basis))
    return B

def proc_action(proc_mat, op_vec, basis):
    '''Get the matrix form of a process matrix times a vectorized operator.

    Parameters
    ----------
    proc_mat: numpy.array
        The process matrix with respect to the given basis
    op_vec: numpy.array
        The vectorized operator with respect to the given basis
    basis: list(numpy.array)
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The matrix form of the image of the vectorized operator under the
        process matrix

    '''
    try:
        output = np.einsum('jk,k,jmn->mn', proc_mat, op_vec, basis)
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        output = np.einsum('jk,k,jmn->mn', proc_mat, op_vec, basis)
    return output

def middle_action(tensor, basis, argument):
    '''Calculate the middle action of the tensor on the argument.

    The middle action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the Kraus-operator way of acting with the tensor:
    B -> A_jk*X_j*B*X^dag_k.

    Parameters
    ----------
    tensor: numpy.array
        The middle-action tensor
    basis: list(numpy.array)
        Basis of operators expressed as matrices
    argument: numpy.array
        The operator to be acted upon

    Returns
    -------
    numpy.array
        The operator resulting from the action

    '''
    try:
        output = np.einsum('jk,jmn,nq,kpq->mp', tensor, basis, argument,
                           np.conj(basis))
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        output = np.einsum('jk,jmn,nq,kpq->mp', tensor, basis, argument,
                           np.conj(basis))
    return output

def left_right_action(tensor, basis, argument):
    '''Calculate the left-right action of the tensor on the argument.

    The left-right action of a tensor A_jk on an operator B, with respect to an
    operator basis {X_j}, is the operator bra-ket way of acting with the tensor:
    B -> A_jk*tr(B*X^dag_k)*X_j.

    Parameters
    ----------
    tensor: numpy.array
        The left-right-action tensor
    basis: list(numpy.array)
        Basis of operators expressed as matrices
    argument: numpy.array
        The operator to be acted upon

    Returns
    -------
    numpy.array
        The operator resulting from the action

    '''
    try:
        output = np.einsum('jk,jmn,kpq,pq->mn', tensor, basis, np.conj(basis),
                           argument)
    except AttributeError:
        # The above won't work if the basis is made of qutip Qobjs
        basis = [np.array(X.data.todense()) for X in basis]
        output = np.einsum('jk,jmn,kpq->pq', tensor, basis, np.conj(basis),
                           argument)
    return output

def get_process_tensor_from_process(process, dim):
    '''Calculate the process tensor given a process.

    The process tensor for a process E in a vector basis {|n>} has the following
    components:
    T_jkmn = <m| E(|j><k|) |n>

    Parameters
    ----------
    process: callable
        The process as a function that takes a density matrix and returns the
        resulting density matrix.
    dim: int
        Dimension of the Hilbert space on which the density operators act

    Returns
    -------
    numpy.array
        The process tensor

    '''
    kets = [np.array([1 if j==k else 0 for k in range(dim)])
            for j in range(dim)]
    rho0_jks = [[np.outer(ket1, ket2.conj()) for ket2 in kets]
                for ket1 in kets]
    rho_jks = [[process(rho0) for rho0 in rho0_ks] for rho0_ks in rho0_jks]
    return np.array(rho_jks)

def act_process_tensor(tensor, state):
    '''Calculate the action of a process tensor on a state.

    The process tensor for a process E in a vector basis {|n>} has the following
    components:
    T_jkmn = <m| E(|j><k|) |n>
    For a state rho = rho_jk |j><k| the action of the process is calculated as
    below:
    E(rho) = T_kjmn rho_jk |m><n|
    so the new density matric elements are given by this particular contraction
    of T with the original density matrix elements.

    Parameters
    ----------
    tensor: np.array
        The process tensor
    state: np.array
        The density matrix

    Returns
    -------
    numpy.array
        The new density matrix to which the process maps the original density
        matrix

    '''
    return np.einsum('kjmn,kj->mn', tensor, state)

def sharp_tensor(A, basis):
    '''Perform the sharp on the tensor with respect to an operator basis.

    The sharp is an involution that swaps the middle and left-right actions of a
    tensor.  The is, the left-right action of A is equal to the middle action of
    A#, and vice versa.

    Parameters
    ----------
    A: numpy.array
        The tensor to sharp
    basis: list(numpy.array)
        Basis of operators expressed as matrices

    Returns
    -------
    numpy.array
        The sharp of the supplied tensor with respect to the given operator
        basis

    '''
    # E = E_jk*|X_j)(X_k|
    # |X_j) = X_j_mn*|m><n|
    # (X_k| = X.conj()_k_qp*|p><q|
    #       = X_k_pq*|p><q| (if Hermitian)
    # E = E_jk*X_j_mn*X_k_qp*|m><n|*|p><q|
    #   = E_mn_pq*|m><n|*|p><q|
    # B_mnpq_jk = X_j_mn*X_k_qp
    # E_mn_pq = E_jk*X_j_mn*X_k_qp
    #         = B_mnpq_jk*E_jk
    # E#_mn_pq = E_mq_pn
    #          = B_mqpn_jk*E_jk
    # E# = E#_mn_pq*|m><n|*|p><q|
    #    = E#_jk*X_j_mn*X_k_qp*|m><n|*|p><q|
    # E#_mn_pq = B_mnpq_jk*E#_jk
    # E#_jk = B.inv()_jk_mnpq*E#_mn_pq
    #       = B.inv()_jk_mnpq*B_mqpn_rs*E_rs
    op_dim = len(basis)
    vec_dim = basis[0].shape[0]
    assert vec_dim**2 == op_dim
    B = get_supdup_op(basis)
    Binv = np.linalg.inv(B.reshape(op_dim**2, op_dim**2)).reshape(
            *it.repeat(op_dim, 2), *it.repeat(vec_dim, 4))
    return np.einsum('jkmnpq,mqpnrs,rs->jk', Binv, B, A)

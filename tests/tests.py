import itertools as it
import numpy as np
from nose.tools import assert_almost_equal, assert_equal, assert_true
import quantum_utilities.supops as supops
from quantum_utilities.qubits import Id, sigx, sigy, sigz, sigp, sigm, ket0, ket1

def sharp_and_dblsharp(A, basis):
    # Calculate the sharp and double sharp of a tensor with respect to a given
    # operator basis.
    A_sharp = supops.sharp_tensor(A, basis)
    A_dblsharp = supops.sharp_tensor(A_sharp, basis)
    return A_sharp, A_dblsharp

def check_sharp_action(A, Asharp, basis, arguments):
    # Make sure the left-right action of the original tensor is equivalent to
    # the middle action of the supplied sharp of that tensor, with respect to
    # the given basis.
    for argument in arguments:
        mid_act = supops.middle_action(Asharp, basis, argument)
        lr_act = supops.left_right_action(A, basis, argument)
        diff_norm = np.linalg.norm(mid_act - lr_act)
        assert_almost_equal(diff_norm, 0.0, 7)

def assert_op_almost_equal(op1, op2, tol=7):
    # Check if two operators are close to one another in norm.
    diff_norm = np.linalg.norm(op1 - op2)
    assert_almost_equal(diff_norm, 0.0, tol)

def test_sharp():
    # Run tests to see if the sharp is behaving properly.

    # Construct Pauli and matrix-unit operator bases for 1 and 2 qubits.
    pauli_basis_1 = [Id, sigx, sigy, sigz]
    unit_basis_1 = [np.outer(ketj, np.conj(ketk))
                    for ketj, ketk in it.product([ket0, ket1],
                                                 repeat=2)]
    pauli_basis_2 = [np.kron(X1, X2)
                     for X1, X2 in it.product(pauli_basis_1, repeat=2)]
    unit_basis_2 = [np.kron(E1, E2)
                    for E1, E2 in it.product(unit_basis_1, repeat=2)]

    RS = np.random.RandomState()

    # Do tests on a real asymmetric single-qubit tensor.
    RS.seed(1904251100)
    A_1 = RS.randn(4, 4)
    A_1_sharp_pauli, A_1_dblsharp_pauli = sharp_and_dblsharp(A_1, pauli_basis_1)
    A_1_sharp_unit, A_1_dblsharp_unit = sharp_and_dblsharp(A_1, unit_basis_1)
    assert_op_almost_equal(A_1, A_1_dblsharp_unit)
    check_sharp_action(A_1, A_1_sharp_pauli, pauli_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1_sharp_pauli, A_1, pauli_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1, A_1_sharp_unit, unit_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1_sharp_unit, A_1, unit_basis_1,
                       pauli_basis_1 + unit_basis_1)

    # Do tests on a real symmetric single-qubit tensor.
    A_symm_1 = A_1 + A_1.T
    A_symm_1_sharp, A_symm_1_dblsharp = sharp_and_dblsharp(A_symm_1,
                                                           pauli_basis_1)
    assert_op_almost_equal(A_symm_1, A_symm_1_dblsharp)
    check_sharp_action(A_symm_1, A_symm_1_sharp, pauli_basis_1, pauli_basis_1)
    check_sharp_action(A_symm_1_sharp, A_symm_1, pauli_basis_1, pauli_basis_1)

    # Do tests on a complex asymmetric single-qubit tensor.
    RS.seed(1904251402)
    A_1_complex = RS.randn(4, 4) + 1j*RS.randn(4, 4)
    A_1_complex_sharp_pauli, A_1_complex_dblsharp_pauli = sharp_and_dblsharp(
            A_1_complex, pauli_basis_1)
    A_1_complex_sharp_unit, A_1_complex_dblsharp_unit = sharp_and_dblsharp(
            A_1_complex, unit_basis_1)
    assert_op_almost_equal(A_1_complex, A_1_complex_dblsharp_unit)
    check_sharp_action(A_1_complex, A_1_complex_sharp_pauli, pauli_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1_complex_sharp_pauli, A_1_complex, pauli_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1_complex, A_1_complex_sharp_unit, unit_basis_1,
                       pauli_basis_1 + unit_basis_1)
    check_sharp_action(A_1_complex_sharp_unit, A_1_complex, unit_basis_1,
                       pauli_basis_1 + unit_basis_1)

    # Do tests on a real asymmetric two-qubit tensor.
    RS.seed(1904251316)
    A_2 = RS.randn(16, 16)
    A_2_sharp, A_2_dblsharp = sharp_and_dblsharp(A_2, pauli_basis_2)
    assert_op_almost_equal(A_2, A_2_dblsharp)
    check_sharp_action(A_2, A_2_sharp, pauli_basis_2, pauli_basis_2)
    check_sharp_action(A_2_sharp, A_2, pauli_basis_2, pauli_basis_2)

    # Do tests on a real symmetric two-qubit tensor.
    A_symm_2 = A_2 + A_2.T
    A_symm_2_sharp, A_symm_2_dblsharp = sharp_and_dblsharp(A_symm_2,
                                                           pauli_basis_2)
    assert_op_almost_equal(A_symm_2, A_symm_2_dblsharp)
    check_sharp_action(A_symm_2, A_symm_2_sharp, pauli_basis_2, pauli_basis_2)
    check_sharp_action(A_symm_2_sharp, A_symm_2, pauli_basis_2, pauli_basis_2)

def check_proc_lr_action(proc_mat, lr_tensor, op_vecs, basis):
    # Make sure the matrix-vector action of the process matrix matches the
    # left-right action of the associated tensor with respect to the given
    # basis.
    proc_acts = [supops.proc_action(proc_mat, op_vec, basis)
                 for op_vec in op_vecs]
    lr_acts = [supops.left_right_action(lr_tensor, basis,
                                        supops.op_vec_to_matrix(op_vec, basis))
               for op_vec in op_vecs]
    for proc_act, lr_act in zip(proc_acts, lr_acts):
        assert_op_almost_equal(proc_act, lr_act)

def test_proc_mat():
    # Run tests to see if the process matrix and left-right tensors are behaving
    # properly with respect to one another.

    # Construct Pauli and matrix-unit operator bases for 1 and 2 qubits.
    pauli_basis_1 = [Id, sigx, sigy, sigz]
    unit_basis_1 = [np.outer(ketj, np.conj(ketk))
                    for ketj, ketk in it.product([ket0, ket1],
                                                 repeat=2)]
    pauli_basis_2 = [np.kron(X1, X2)
                     for X1, X2 in it.product(pauli_basis_1, repeat=2)]
    unit_basis_2 = [np.kron(E1, E2)
                    for E1, E2 in it.product(unit_basis_1, repeat=2)]
    RS = np.random.RandomState()

    # Do tests on a real asymmetric single-qubit process matrix.
    RS.seed(1904251318)
    op_vecs_1 = RS.randn(4, 4)
    RS.seed(1904251319)
    proc_mat_1 = RS.randn(4, 4)
    lr_tensor_1_pauli = supops.proc_mat_to_LR_tensor(proc_mat_1,
                                                     pauli_basis_1)
    check_proc_lr_action(proc_mat_1, lr_tensor_1_pauli, op_vecs_1,
                         pauli_basis_1)
    lr_tensor_1_unit = supops.proc_mat_to_LR_tensor(proc_mat_1,
                                                    unit_basis_1)
    check_proc_lr_action(proc_mat_1, lr_tensor_1_unit, op_vecs_1,
                         unit_basis_1)

    # Do tests on a complex asymmetric single-qubit process matrix.
    RS.seed(1904251319)
    proc_mat_1_complex = RS.randn(4, 4) + 1j*RS.randn(4, 4)
    lr_tensor_1_complex_pauli = supops.proc_mat_to_LR_tensor(
            proc_mat_1_complex, pauli_basis_1)
    check_proc_lr_action(proc_mat_1_complex, lr_tensor_1_complex_pauli,
            op_vecs_1, pauli_basis_1)
    lr_tensor_1_complex_unit = supops.proc_mat_to_LR_tensor(
            proc_mat_1_complex, unit_basis_1)
    check_proc_lr_action(proc_mat_1_complex, lr_tensor_1_complex_unit,
                         op_vecs_1, unit_basis_1)

    # Do tests on a complex asymmetric two-qubit process matrix.
    RS.seed(1904251317)
    op_vecs_2 = RS.randn(16, 16)
    RS.seed(1904251320)
    proc_mat_2 = RS.randn(16, 16)
    lr_tensor_2_pauli = supops.proc_mat_to_LR_tensor(proc_mat_2,
                                                     pauli_basis_2)
    check_proc_lr_action(proc_mat_2, lr_tensor_2_pauli, op_vecs_2,
                         pauli_basis_2)
    lr_tensor_2_unit = supops.proc_mat_to_LR_tensor(proc_mat_2,
                                                    unit_basis_2)
    check_proc_lr_action(proc_mat_2, lr_tensor_2_unit, op_vecs_2,
                         unit_basis_2)

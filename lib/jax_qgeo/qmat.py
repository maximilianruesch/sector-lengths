from __future__ import division, print_function

import scipy as sp
import jax.numpy as np
import itertools as it
from scipy.linalg import det

#### basic Matrix functions (for Hermitian operators) ####

def dim(A):
    """ returns dimension of Hilbert space, in which operator A lives """
    return np.int(np.rint((np.shape(A)[0])))

def lambda_min(A):
    return sp.linalg.eigvalsh(A, eigvals=(0,0))[0]

def lambda_max(A):
    N = dim(A)
    return sp.linalg.eigvalsh(A, eigvals=(N-1, N-1))[0]

def dag(A):
    """ hermitian, dagger of operator A
        args:       A : ndarray
        returns:    ndarray, hermitian of array
    """
    return np.conjugate(np.transpose(A))

def vec_to_mat(vec):
    if np.ndim(vec) == 1:      # if input is a ket
        return np.outer(vec, dag(vec))
    else: # do nothing if input is already a density matrix
        return vec

def msqrt(A):
    """ matrix root
        for positive matrix A, positive matrix is always hermitian
    args:       A : ndarray
    returns:    ndarray
    """
    assert is_positive(A)
    [ev, P] = sp.linalg.eigh(A)
    D_sqrt = np.diag(np.sqrt(ev))  # square root of diagonal
    # return sp.linalg.fractional_matrix_power(A,1/2)    #slower, more exact
    return np.dot(P, np.dot(D_sqrt, dag(P)))  # faster but less exact

def mpow(A, k):
    """ matrix power """
    return sp.linalg.fractional_matrix_power(A, k)

def mabs(A):
    """ matrix absolute value, returns positive operator
        as defined in BZ 8.12 |A| = \sqrt(A \dag(A))
        for arbitrary matrix A
    args:       A : ndarray
    returns:    ndarray
    """
    # careful: order of A & A^dag, positivity of eigenvalues, correct definitions...
    return sp.linalg.sqrtm(np.dot(A, dag(A)))
    
def mdot(op_list):
    """ multiple dot product, takes dot product of all operators in list
    args: list of ndarray
    return: ndarray
    """
    if len(op_list) == 1:
        return op_list[0]
    return np.linalg.multi_dot(op_list)

    
def mdot_self(op, k):
    if k == 0:
        D = dim(op)
        return np.eye(D)
    out = op
    for it in range(k-1):
        out = np.dot(out, op)
    return out

#### Matrix Checks ####

def is_positive(A, tol=1e-14):
    """ returns true if A is positive semidefinite """
    return sp.linalg.eigvalsh(A, eigvals=(0,0))[0] >= - tol

def is_psd(A, tol=1e-14):
    return is_positive(A, tol=1e-14)

def is_herm(A):
    return np.allclose(dag(A), A)

def is_unitary(U, tol=1e-14):
    D = np.shape(U)[0]
    return np.linalg.norm( np.dot( np.transpose(U).conjugate(), U)  - np.eye(D)) < tol

### tensor stuff
#def pad_single_op(X, k, d=2):
#    """ pads operator to be at position k with identities on
#        the remaining subsystems"""
#    return tensor_mix
    

# Multilinear algebra

def permanent(a):
    m = np.shape(a)[0]
    perm = 0
    for p in it.permutations(range(m)):
        perm = perm + np.product([a[i,p[i]] for i in range(m)])
    return perm

def diag_prod(a):
    return np.product(np.diag(a))
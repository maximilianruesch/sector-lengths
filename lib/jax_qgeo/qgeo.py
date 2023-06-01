#! /usr/bin/python

# Author:     Felix Huber, University Siegen
#             Nikolai Wyderka, University Siegen
# Project:    Quantum Information, Quantum Geometry
# Module:     Main library

"""
what it does:
- generate, manipulate, and analyse random/special quantum states with an 
emphasis on multipartite qubit states, Bloch decomposition, and graph states.

- Support for operations on higher- and mixed-dimensional states, such as 
  partial trace, partial transpose, splicing of operators (tensor_mix)
e.g.
- generate multi qubit states, obtain and manipulate Bloch/Pauli coefficients
- generate (k-local) pauli basis
- generate haar-random unitaries, states, density matrices

- transform states ket <-> density matrix, tests if proper state, etc
- generate and manipulate (hyper-) gaph states, fast partial traces for graph 
  states
- calculate various distances between states
- calculate ground/thermal states, gap, degeneracy, etc

Some more project related stuff
- information projection: routine, taking into account symmetries
- AMES/k-uniform routines: functions to obtains sector lenghts
  k-uniform state search
- error correction checks: Knill Laflamme kriterion

Note:
most functions operate on nxn or nx1 numpy arrays, representing kets or density 
matrices 
    (variables named as 'rho', 'sigma', 'A' or 'psi')
some functions operate on dictionaries, corresponding to Pauli decompositions
    (variables often named as 'coeff')
some functions operate on strings, e.g. corresponding to states or a basis.
    (variables or functions named like 'function_s')

some functions are named like ket_do_thing, dm_do_thing, op_do_thing, 
    coeff_do_thing, do_thing_from_that
    such that it is clear what they act on.

Definitions from  - Bengtsson & Zyckowski, Geometry of Quantum states
                  - Guehne & Toth, Entanglement detection
                  - references to papers are given
"""

from __future__ import division, print_function

import jax.numpy as np
import numpy
from jax.numpy import dot
import scipy as sp
import scipy.linalg

from scipy import sparse
from math import sqrt, pi

import math as mt
#import string as st
import random as rnd
import itertools as itt  # for combinatorics, choice, permutations, etc
import time
import re
from fractions import Fraction

#from . import qmat as qm
from .qmat import dag, is_herm, vec_to_mat, mdot, lambda_min, lambda_max, mabs, msqrt,mpow
#from qmat import dag, is_herm, vec_to_mat, mdot, lambda_min, lambda_max, mabs, msqrt,mpow

def try_import_qecc():
    """ tries to import qecc library """
    try:
        import qecc as q
    except: 
        print('QuaEC: Quantum Error Correction Analysis in Python not installed.'
              'goto http://www.cgranade.com/python-quaec/')
    return q

# optimised numpy replacement (can be removed if in doubt/circumstances change)

def trace(A):
    """ returns trace of operator A, 4x faster than trace() """
    return np.einsum('ii->', A)

def single_ket(j, d=2):
    """ ket vector for single d-dim system
        single_ket(3, d=5) returns |3>
        args: str, e.g. q = 2, d=3
        returns: ndarray, ket
    """
    assert 0 <= j < d
    s = np.zeros(d)
    s[j] = 1
    return s


qb = dict()
qb['0'] = np.array([1, 0])
qb['1'] = np.array([0, 1])
qb['+'] = np.array([1, 1]) / sqrt(2)
qb['-'] = np.array([1,-1]) / sqrt(2)

def single_qbket(state):
    """ produces |0>, |1>, |+>, |-> kets, needed for function ket above.
        returns: ndarray,  0 or 1 ket. usage: single_qbket('0')
    """
    return qb[state]

def ket(q_string, d=2):
    """ general ket function for d-dimensional systems (d <= 9)
        args: str, e.g. '0100' or '0230', d=4
        returns: ndarray, ket
    """
    if d == 2:
        return qbket(q_string)
    else:    
        return mkron([single_ket(int(el), d) for el in q_string])


def nket(j,d=2):
    """ numerical ket, accepting a single number as input
        returns: array, |j mod d>   
    """
    return ket(str(j%d), d)

def qbket(qubit_string):
    """ given string of 0, 1, +, -, returns associated ket 
        args: qubit_string : str    e.g. '0111', '00+1--'
        returns: ndarray : tensor product of above representation    
    """
    qubit_list = list(qubit_string)
    ## apply ket_1b function on each elemnt in qubit_list
    qubit_list_matrix = list(map(single_qbket, qubit_list) )
    return mkron(qubit_list_matrix)


def ket_from_list(ket_list, coeffs):
    """ given list of kets (as strings) and list of corresponding coefficients
        constructs normalised state.
    """    
    s = np.zeros(2**len(ket_list[0]))
    for it in range(len(ket_list)):
        s = s +  coeffs[it] * ket(ket_list[it])
    return ket_normalise(s)

def dm(state):
    """ given a ket, returns it's associated density matrix (outer product)
        args:      ndarray, vector/ket, e.g. ket('01')
        returns:    ndarray, density matrix
    """
    return vec_to_mat(state)

def make_dm(state):
    """ wrapper for dm. will make ket to density matrix if not already"""
    return dm(state)
    
def to_ket(state):
    """ transforms a pure state into a ket """
    if np.ndim(state) == 2: # if input is a dm
        evl, psi = sp.linalg.eigh(-state, eigvals = (0,0))    
        return psi[:,0]
    else:
        return state

def to_sparse(A):
    """ returns matrix as sparse matrix in coo format """
    if sp.sparse.issparse(A) == True:
        return A
    else:
        return sp.sparse.coo_matrix(A)


def ket_normalise(phi):
    """ normalise ket """
    return phi / sqrt(np.abs(dot(dag(phi), phi)))

def trans_prob(psi, phi):
    """ transition probability between psi and phi"""
    return (np.abs(braket(psi, phi)))**2


def braket(alpha, beta):
    """ returns inner product of psi and phi 
        args:       psi, phi : ndarray, vector    
        returns:    scalar, inner product
    """
    return dot( dag(alpha), beta )
    
def ketbra(alpha, beta, d=2):
    """ outer product for vectors.
        also accepts strings and optional dimension: 
        '01', '11'  ->  |01><11|    
    """
    if type(alpha) == str:
            alpha = ket(alpha, d)
            beta  = ket(beta, d)
    return np.outer(alpha, dag(beta))

def HS_inner(A,B):
    """ Hilbert-Schmidt inner product """
    return trace( np.dot(dag(A), B) )

### checks on states
    
def is_state(state, tol= 1e-14):
    """ returns True if it is a proper ket / density matrix
        checks trace one, hermiticity, positivity
        args: ndarray, ket or density matrix
        returns: bool, True if it is a state
    """
    rho = make_dm(state)
    return np.isclose(1, np.trace(rho), atol=tol)  \
            and is_herm(rho)                    \
            and lambda_min(rho)  >= -tol

def is_pure_state(rho, tol=1e-14):
    """ returns True if it is a proper pure ket / density matrix
        args: ndarray, ket or density matrix
        returns: bool, True if it is a state    
    """
    rho = make_dm(rho)
    return is_state(rho, tol=tol) and lambda_max(rho) >= 1-tol    # purity 1

def is_ket(state, tol=1e-14):
    """ returns True if it is a ket
        args: ndarray, ket
        returns: bool, True if it is a ket
    """
    return np.ndim(state) == 1 and is_pure_state(state, tol)

def is_dm(state, tol=1e-14):
    """ returns True if it is a proper density matrix (not a ket)
        args: ndarray, density matrix
        returns: bool, True if it is a state    
    """
    return np.ndim(state) == 2 and is_state(state, tol)


### Entanglement tests
    
def PPT_crit(rho, d=2, eps=1e-16):
    """ returns true and the corresponding NPT subsystem 
        if the state is entangled, 
        i.e. if state has a negative partial transpose 
        smaller than -eps across some partition
    """
    n = number_of_qudits(rho, d)
    
    for sub in subsets(range(n)):
        if lambda_min(ptranspose(rho, sub, d)) < -eps:
            return True, sub
    return False, None    


### Pauli Tools

# Pauli & Gates
I = np.array([[1, 0], [0, 1]]);     X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]]);  Z = np.array([[1, 0], [0, -1]])
H = np.array([[1,1],[1,-1]])
phase = np.array([[1, 0], [0, 1j]])

Is = sparse.coo_matrix(I); Xs = sparse.coo_matrix(X)
Ys = sparse.coo_matrix(Y); Zs = sparse.coo_matrix(Z)


def P(pauli_str):
    """ tensor product of Paulis in string, e.g. P('IIXZY')
        special behaviour: P('') gives 1, 
                           '-XZZI' yiels an additional overall minus sign
        
        arg:    str : containing -, I, X, Y, Z; e.g. '-XIIZ'
        return: ndarray : density matrix representing multi qubit system
    """
    c=1
    if pauli_str[0] == '-':
        c = -1
        pauli_str = pauli_str[1:]
                
    pauli_l = list(pauli_str)                       # string to list
    # evaluate symbols. Equivalent to map(P, pauli_l), but 30% faster
    pauli_l_matrix = [eval(el) for el in pauli_l]
    return c * mkronf(pauli_l_matrix)                    # tensoring


def P_loc(Pauli_op, idx, n):
    """ tensor product of Pauli_op at locations specified in idx
        arg:        Pauli_op (string) e.g. X,Y,Z
                    idx (list), e.g. [0,3]
                    n (int), number of subsystems
        return:     ndarray : density matrix representing multi qubit system
    """
    l = [('X' if el in idx else 'I') for el in range(n) ]
    s = ''.join(l)
    return P(s)


def P_sparse(pauli_str):
    """ sparse tensor product of Paulis in string
    arg:    str : containing I, X, Y, Z; e.g. 'XIIZ'
    return: ndarray : sparse density matrix representing multi qubit system
            special behaviour: P('') gives 1
    """
    pauli_l = list(pauli_str)                       # string to list
    pauli_l_matrix = [eval(el+'s') for el in pauli_l]
    return mkrons(pauli_l_matrix)                    # tensoring

def P_sym(s, sym=None):
    """ returns symmetrised pauli product of string, 
        according to permutation symmetry of subsystems
        args:   s : string, e.g. 'ZXIII'
                sym : list, permutation symmetry of subsystems, e.g. [0,1,4]
    """
    if P_sym == None:
        return P(s)
    p_arr = np.array(list(s))
    p_arr_s = p_arr[sym] # extract elements to be permuted (subsystem)
    # list of permutations
    p_arr_s_perm = sort_unique(list(itt.permutations(p_arr_s)))
    l = list()
    # loop through permutations of specific symmetry
    for perm_s in p_arr_s_perm: 
        p_arr[sym] = perm_s       # insert permutation back into subsystems
        #print(type(p_arr))
        li = ''.join(list(p_arr)) # make back into string and wire together
        l.append(li) 
    return P_nl(l)

def ket_perm_u(s):
    """ returns sum of tensor products of unique permutations of ket-string"""
    l = string_permutations_unique(s)
    return np.sum([ket(el) for el in l],0)
    
def ket_cycle(s):
    """ returns sum of tensor products of (non-unique) cyclic permutations 
        of ket-string
        arg: s, string
        returns: array, ket
    """
    return np.sum([ket(string_shift(s, it)) for it in range(len(s))], 0)  

def ket_shift(s, shift):
    """ returns sum of tensor products of (non-unique) cyclic permutations 
        of ket-string
        arg: s, string
        returns: array, ket
    """
    return ket(string_shift(s, shift))


def P_perm_u(s):
    """ returns sum of tensor products of unique permutations of Pauli-string"""
    l = string_permutations_unique(s)
    return np.sum([P(el) for el in l],0)    
    
def P_cyclic(pauli_str):
    """ out of 'XZI', returns sum of tensor products of all cyclic permutations 
        XZI, IXZ, ZIX.
        args:       pauli_string : str, e.g. 'ZIII'
        returns:    ndarray with dim 2**n
    """
    n = len(pauli_str)
    pauli_doublestr = 2 * pauli_str
    H = np.zeros(2**n)
    for k in range(n):
        H =  H + P(pauli_doublestr[k:k + n])
    return H

## Gell Mann Basis
  
def GM_X(k, l, d=2):
    """ higher dim Y Gell Mann """
    assert k < l
    Xgen = np.zeros([d, d], dtype=complex)
    Xgen[l,k] =  1
    Xgen[k,l] =  1
    return Xgen #sqrt(D/2) * 

def GM_Y(k, l, d=2):
    """ higher dim Y Gell Mann """
    assert k < l
    Ygen = np.zeros([d, d], dtype=complex)
    Ygen[l,k] =  1j
    Ygen[k,l] = -1j
    return Ygen #sqrt(D/2) * 

def GM_kl(k,l,d=2):
    if k < l:
        return GM_X(k,l,d)
    elif k > l:
        return GM_Y(l,k,d)
    elif k == 0:
        return Id(1,d=d)
    elif k != d-1:
        return direct_sum(GM_kl(k,k,d-1), np.array([[0]]))
    else:
        return np.sqrt(2.0/(d*(d-1)))*direct_sum(GM_kl(0,0,d-1), np.array([[1.0-d]]))

def GM(l, d=2):
    idxs = sorted([(i,j) for i in range(d) for j in range(d)], key=lambda x: 9999*sum(x)+999*max(x)+d*x[0]+x[1])
    if l != 0:
        return GM_kl(idxs[l][0], idxs[l][1], d=d)/sqrt(2.0/d)
    return GM_kl(idxs[l][0], idxs[l][1], d=d)


### Heisenberg-Weyl Basis

def HW_X(D,k=1):
    """ k-th power of Heisenberg Weyl X (shift) matrix 
        Eq 15 in Scott, arXiv:quant-ph/0310137
    """
    return off_diagonal(D, -k) + off_diagonal(D, D-k)

def HW_Z(D, k=1):
    """ k-th power of Heisenberg Weyl Z (clock) matrix 
        Eq 15 in Scott, arXiv:quant-ph/0310137
    """
    return np.diag(np.exp(2j*pi*k*np.arange(D)/D) )

def HW(mu, nu, d=2):
    """ Heisenberg-Weyl / displacement operator basis
        $e^{\pi i \mu \nu/D} X^\mu Z^\nu$
        Eq 14 in Scott, arXiv:quant-ph/0310137
    """
    return np.exp(1j * pi * mu * nu / d) * mdot([HW_X(d, mu), HW_Z(d, nu)])


### Support and Dimensionality

def number_of_qubits(A):
    """ returns number of qubits"""
    return to_int(np.log2(np.shape(A)[0]))

def number_of_qudits(A, d=2):
    """ returns number of qudits (d-level)"""
    k = np.shape(A)[0]
    return to_int( np.log(k)/np.log(d) )

def weight(op):
    """ given operator, returns number of qubits where it acts on nontrivially 
        args:       op : ndarray
        returns:    int : weight of operator 
    """
    coeff = pauli_coeff(op)
    coeff_not_zero =  dict((k, v) for k, v in coeff.items() if v != 0)
    w = 0
    for s in coeff_not_zero.keys():
        w = max(w, len(s) - s.count('I'))
    return to_int(w)    

def support_str(s):
    """ given pauli string (e.g. 'IXZZIY'), returns indices of all nontrivial terms """
    Xs = list(find_letter_in_string(s, 'X'))
    Ys = list(find_letter_in_string(s, 'Y'))
    Zs = list(find_letter_in_string(s, 'Z'))
    ind = Xs + Ys + Zs
    return np.sort(ind)
    
def shorten_to_non_triv_s(s):
    """ removes identity ('I') in string s """
    return ''.join(c for c in s if c in ['X', 'Y', 'Z'])

def weight_str(s):
    """ # of non-trivial (non-identity) terms in Pauli string """
    return len(s) - s.count('I')


#### PAULI FUNCTIONALITY ####

def pauli_basis_str(n):
    """ generator: yields strings representing full basis for n qubits
        access all by list(pauli_basis_str(n))
    """
    for k in range(4**n):
        yield quad2pauli(dec2quad(k, n))

def pauli_basis_k_local_str(n, k_vec):
    """ returns list of strings representing elements of pauli basis for n qubits,
        where each element acts on exactly k qubits nontrivial
        if k is a vector, returns all interactions with order as found in k-vector.
        k : array, locality to be returned        
        
        e.g. all interactions up to k: pauli_basis_k_local_str(n, np.arange(0,k+1))
        ### fixme: make consistent with pauli_basis_str

    """
    if type(k_vec) == int or type(k_vec) == np.int64:
        k_vec = [k_vec] # transform into a list
        
    B = pauli_basis_str(n)

    # case 1: return exactly k-local terms
    if len(k_vec) == 1:
        Bred = [ el for el in B if weight_str(el) == k_vec[0] ]

    # case 2: return terms corresponding to elements in k-vector
    else:
        Bred = list() # reduced basis
        for k_el in k_vec:
            Bred = Bred + pauli_basis_k_local_str(n, [k_el] )
    return Bred


def pauli_basis(n, k_vec = None, sparse=False):
    """ returns generetor of (k-local) Pauli_basis 
        n : int, number of qubits
        k : array, locality to be returned
    """
    if k_vec == None:
        B_str = pauli_basis_str(n)
    else:
        B_str = pauli_basis_k_local_str(n, k_vec)
    if sparse == True:
        return itt.imap(P_sparse, B_str) # applies P() function on each element of B_str    
    else:
        return itt.imap(P, B_str) # applies P() function on each element of B_str


### restricted sets of pauli basis - according to graph

def idx_nonzero(adjM):
    """ get indices of nonzero elements, needed for pauli_basis_str_graph
        ### TODO: refactor name
    """
    idx = np.nonzero(adjM)
    return zip( idx[0], idx[1]) # get tuple of indices

def pauli_basis_str_adj(adjM):
    """ 2-local pauli basis according to graph """
    n = np.shape(adjM)[0]
    B = pauli_basis_k_local_str(n, [2])
    
    # generate interaction positions/indices for all pauli_basis elements
    l = len(B)
    Bc = list(B)
    for k in range(l):
        # replace non-trivial interactions by Q
        Bc[k] = str.replace(Bc[k], 'X', 's')
        Bc[k] = str.replace(Bc[k], 'Y', 's')
        Bc[k] = str.replace(Bc[k], 'Z', 's')
    
    # indices from pauli basis
    B_idx = []
    for k in range(l):
        # find positions of both Qs in interactions
        B_idx.append((str.find(Bc[k], 's'), str.rfind(Bc[k], 's')))

    # indices from adjacency matrix
    adj_idx = idx_nonzero(adjM)

    # compare with indices obtained from adjacency matrix   
    B_graph = []
    for k in range(l):
        if B_idx[k] in adj_idx:
            B_graph.append(B[k])
    return pauli_basis_k_local_str(n, [0,1]) + B_graph # add 0 & 1-local interactions, too


def pauli_basis_adj(adjM):
    """ Pauli basis of adjacency matrix """
    B_str = pauli_basis_str_adj(adjM)
    return itt.imap(P, B_str) # applies P() function on each element of B_str

####

def dec2quad(x, n):
    """ turns base 10 x into base 4 x, filling on the left until width is n digits
        to be used in conjuction with quad2pauli
    """
    s = ''
    if x == 0:
        return str.zfill(s, n)   # pad on the left with zeros
    while x > 0:
        s = str(x %  4) + s         # modulo
        x = x // 4                  # integer division
    return str.zfill(s, n)

def dec2base(x, q, n):
    """ turns base 10 x into base q x, filling on the left until width is n digits
        to be used in conjuction with quad2pauli
    """
    s = ''
    if x == 0:
        return str.zfill(s, n)   # pad on the left with zeros
    while x > 0:
        s = str(x %  q) + s         # modulo
        x = x // q                  # integer division
    return str.zfill(s, n)


def quad2pauli(quadnumber):
    """ change string of numbers in base 4 (0,1,2,3) into string of pauli I,X,Y,Z
        usage: quad2pauli('0123') = 'IXYZ'
    """
    quadnumber      = str.replace(quadnumber, '0', 'I')
    quadnumber      = str.replace(quadnumber, '1', 'X')
    quadnumber      = str.replace(quadnumber, '2', 'Y')
    pauli_string    = str.replace(quadnumber, '3', 'Z')
    return pauli_string    


def pauli_coeff(A, k=None, herm=True):
    """ returns dictionary of coefficients in Pauli basis,
        for Hermitean operators
        args:   
            A :  ndarray, operator/density matrix of n-qubit system
            k : array/list, locality of coefficients to be returned
            herm : bool, true if operator is Hermitean 
                      (yields real expectation values.)
        returns:
            dict, coefficients in Pauli basis
    """
    A = dm(A)
    n = number_of_qubits(A)
    
    c = dict()
    if k is None:
        B = pauli_basis_str(n)
    else:
        B = pauli_basis_k_local_str(n, k)
    for el in B:
        c[el] = expect_fast(A, el, herm=herm)
    return c


def pauli_coeff_gen(A, k = None, herm=True):
    """" generator, returns pauli coefficients """
    n = number_of_qubits(A)
       
    if k is None:
        B = pauli_basis_str(n)
    else:
        B = pauli_basis_k_local_str(n, k)
    for el in B:
        yield el, expect_fast(A, el, herm=herm)
        
    
def pauli_coeff_adj(A, adjM):
    """ returns dictionary of coefficients in restricted pauli basis
        whose 2-local interactions have the structure of adjacency matrix adjM
    """        
    # write tests<
    c = dict()
    # normalisation such that rho == \sum_i Tr(B_i*A)*B_i
    # if B_i is a complete basis    
    for el in pauli_basis_str_adj(adjM):
        c[el] = trace(dot(P(el), A)).real
    return c

def pprint_pauli_coeff(A, k=None, herm=True, tol=1e-10, mode=0):
    """ print dictionary of Pauli nonzero coefficients of hermitian operators
        if k_vec provided, only prints coefficients with #nontrivial terms = k
        args:   A, B :  ndarray
                k : list of int, weights to pprint
                herm : bool, true if operator is Hermitean 
                      (yields real expectation values.)
                tol : tolerance, below which coefficients are not displayed
                mode : mode 0 prints as dictionary, mode 1 prints as sum over Paulis
        returns: prints dictionary of pauli coefficients
    """

    A = make_dm(A)
    cA = pauli_coeff(A, k, herm=herm)

    # rounding
    for k in cA.keys():
        digits = to_int(np.log10(1/tol))
        cA[k] = np.around(cA[k], digits)

    # dumping zero values
    c_not_zero =  dict((k, v) for k, v in cA.items() if abs(v) > tol)

    if mode==0:     # dictionary pprint
        from pprint import pprint
        pprint(c_not_zero)
        return
    
    elif mode==1: # "math mode", prints paulis, rounds, omits 1.0 prefactors
        # replacing it with signs.
        for (k, v) in c_not_zero.items():
            if np.round(np.abs(v), 10) == 1:
                if np.sign(v) == 1.0:
                    v_mod = '+' 
                elif np.sign(v) == -1.0:
                    v_mod = '-' 
            else:
                v_mod = np.round(v, 4)
            print(v_mod, k)
    return
        
def reconstruct_operator(coeff, n):
    """ reconstructs operator from given coefficients in dictionary, for system with n qubits"""
    "slightly slower but shorter: np.sum( [value * P(key) for key, value in c.items()], 0) "
    
    A = np.zeros((2**n, 2**n))
    for key, value in coeff.items():
        A = A + value * P(key)
    return A / 2**n #normalisation
    
def op_local(A, kk):
    """ removes interactions of different order than kk 
        args: A, ndarray, operator
        kk: list of ints, e.g. [0,1,2]
        returns: ndarray of same shape as A
    """
    n = number_of_qubits(A)
    c = pauli_coeff(A, kk)
    return reconstruct_operator(c, n)
    
def coeff_k_local(coeff, n, k_vec):
    """ returns coefficients corresponding to k-local interactions in dict of n qubits
    param:  coeff : dictionary
            n, k  : integer
    returns: dict : (string, float), dictionary of Pauli strings and corresponding coefficients
    """
    # todo move to qgeo

    if type(k_vec) == int:
        k_vec = [k_vec] # transform to list

    # case 1: return exactly k-local & non-vanishing terms
    if len(k_vec) == 1:
        coeff_out = dict((k, v) for k, v in coeff.items() if weight_str(k) == k_vec[0] \
        and v != 0 )

    # case 2: return terms corresponding to elements in k-vector
    else:
        coeff_out = dict()
        for k_el in k_vec:
            coeff_out.update(coeff_k_local(coeff, n, [k_el] ))
    return coeff_out


#### GRAPH STATES ####

def graph_state_from_adj(adj):
    """ create graph state from adjacency matrix 
        arg:     adjacency matrix, n x n ndarray    
        returns: ket, ndarry    
    """
    edges = adj_to_edge_set(adj)
    n = np.shape(adj)[0]
    return graph_state_from_edge_set(n, edges)

    
def adj_to_generator(adj):
    """ graph state functionality: graph states, returns generator for given 
    adjacency matrix. not for hypergraph states!
        args: adj : ndarray, adjacency matrix
        return: list of strings, e.g. ['ZZI', 'IZZ']
    """
    d = np.shape(adj)[0]
    G = ['']*d
    for j in range(d):
        row = list(adj[j,:])
        for k in range(d):
            if k == j:
                row[k] = 'X'
            elif row[k] == 1:
                row[k] = 'Z'
            elif row[k] == 0:
                row[k] = 'I'
        G[j] = ''.join(row)
    return G

def adj_to_edge_set(adj):
    """ graph state functionality: creates edge set from adjacency matrix 
        args: adj : ndarray, adjacency matrx
        returns: list of list, edge set.
    """
    l = map(list, idx_nonzero(adj))
    l = map(sorted, l)
    return list(sort_unique(l))

def edge_set_to_adj(n, edge_set):
    """ given edge_set, returns adjacency matrix
        edge_set, list of lists, e.g.[[0,1], [2,3]]
        n : int, number of qubits
    """
    adj = np.zeros([n,n])
    for edge in edge_set:
        if len(edge)>=3 or len(edge) <=1:
            print('hypergraph or 0-edges present')
        adj[edge[0], edge[1]] += 1
    adj = adj + np.transpose(adj)
    return adj

def edge_set_to_generator(n, edge_set):
    adj = edge_set_to_adj(n, edge_set)
    return adj_to_generator(adj)


def stabiliser(G):
    """ returns stabiliser given generator as pauli elements
        arg: G : list, e.g. ['ZXZI', 'IZXZ']
        returns: list of array containing stabiliser elements
    """
    n = len(G[0])
    G_set = set(G)
    pow_set = set(powerset(G_set))
    pow_set.remove(()) # remove empty element
    for el in pow_set:
        yield mdot(map(P, el))
    yield np.eye(2**n)


# code dependent on qecc / QuaEC, http://www.cgranade.com/python-quaec/
def stabiliser_str(G):
    """ returns stabiliser given generator as pauli elements
        arg: G : list, e.g. ['ZZI', 'IZZ']
        returns: list containing stabiliser elements as strings [III, IZZ, ZZI, ZIZ]
    """
    q = try_import_qecc()

    stab = q.StabilizerCode(G, [], [])
    return stab.stabilizer_group()


def min_weight_stabiliser(G):
    """ returns minimum weight of any element in stabiliser, given generator 
        also works on very big stabilisers.
        arg: G, generator, e.g. ['ZZI', 'IZZ']
        returns: int, minimum weight of elements in stabiliser (apart from identity)
    """
    q = try_import_qecc()

    stab = q.StabilizerCode(G, [], [])
    S = stab.stabilizer_group()
    S.next() # skip first (identity) element
    delta = np.inf
    for el in S:
        delta = min(delta, el.wt)
    return delta
    
def same_stabiliser(G1, G2):
    """ tests if to generators correspond to the same stabiliser 
        (compares stabiliser sets)    
    """
    q = try_import_qecc()
    
    stab1 = q.StabilizerCode(G1, [], [])
    stab2 = q.StabilizerCode(G2, [], [])
    return  set(stab1.stabilizer_group()) == set(stab2.stabilizer_group()) 


#### (HYPER-)GRAPH STATE FUNCTIONALITY ####

def P_nl(nl_list):
    """ "non-local" Pauli operator,  needed for nonlocal stabiliser / hypergraph states
        args: list of str, e.g. ['ZZI', 'IYI']        
        returns: sum of tensor products of el in nl_list
    """
    if len(nl_list) == 1:
        return P(nl_list[0])        
    elif type(nl_list) == str:
        return P(nl_list)
    else:
        return np.sum([P(el) for el in nl_list], axis=0)


# TODO: rename functions.

def graph_state(G,n=None):
    """ alias"""
    return stabilizer_state(G, n=None)

def stabilizer_state(G, n=None):
    """ calculate stabilizer state from generator
        arg: list of generators, e.g. ['ZXZI', 'IZXZ', 'ZIZX', 'XZIZ']
    """
    if n == None:
        n = len(G[0])
    Id= np.eye(2**n)
    R = np.eye(2**n)
    for g in G:
        R = dot(R, (P_nl(g) + Id)) ### P_nl instead of P for non-local generator
    return R / trace(R)

def hypergraph_state(n, edges):
    """ wrapper for graph_state_from_edge_set()
        create hypergraph state of n qubits with edges as specified in edge set,
        e.g. [[0,1,2],[3,6,5]]
    """
    return graph_state_from_edge_set(n, edges)

def graph_state_from_edge_set(n, edges):
    """ creates (hyper-) graph state (ket) from edge set
        each edge in the edge set has to be unique!
        args:    n, int, number of subsystems                 
                 edges, list of lists, edge set. e.g. [[0,1,2],[3,6,5]]
                 has to be non-empty!
                 not yet implemented: d, int, dimensionality of subsystems  #fixme: implement dimensionality
        returns: ket, ndarray of len 2**N
    """
    #TODO. implement for qudits
    #Create equal superposition
    N = 2**n
    vec = np.ones(N) # |+> ^n  state
    
    #Create binary representation of state
    powers_of_two = 2** ( np.arange(n)[::-1] )
    binary_rep    = (np.arange(N)[:, np.newaxis] & powers_of_two) / powers_of_two
    
    #apply edges. e.g. 2-edge: C_phase = |00><00| + |01><01| + |10><10| - |11><11| = diag(1,1,1,-1)
    for edge in edges:
        for it_2 in range(len(binary_rep)):
            a = binary_rep[it_2]
            # relevant elements of vector, should be all 1 to multiply vec by -1
            if sum(a[edge]) == len(edge):
                vec[it_2] *= -1
    return vec / sqrt(N)

def hypergraph_count_edges(h_list, k=None):
    """ given edgelist, counts edges (not counting 0- and 1-edges) """
    c = 0
    if k == None:
        for el in h_list:
            if len(el)>=2:
                c+=1
    else:
        for el in h_list:
            if len(el) == k:
                c+=1
    return c

def rand_hypergraph_state(n):
    """ returns random hypergraph state and corresponding edges, 
        i.e. states with equal entries having random signs 
        args: n : int,  number of qubits
        returns: psi, ndarray, ket
                 hh, list of list, edge_set
    """
    N = 2**n 
    psi = np.random.choice([-1,1], size = N) / sqrt(N)
    hh = graph_state_to_edge_set(psi)
    return psi, hh

def rand_graph_state(n):
    """ returns random graph state and edges, 
        (2-edges only)
        args: n : int,  number of qubits
        returns: psi, ndarray, ket
                 hh, list of list, edge_set
    """
    psi, hh = rand_hypergraph_state(n)
    hh2 = [h for h in hh if len(h) == 2]
    psi2 = hypergraph_state(n, hh2)
    return psi2, hh2
    

def graph_state_to_edge_set(psi):
    """ Given (hyper-) graph state psi, return edge-set
        args: psi : ndarray, ket
        returns: list of list of integers, edge set
    """
    n = number_of_qubits(psi)
    try:
        v = psi / np.abs(psi)
    except: 'not a hypergraph state'
    edge_set = []
    for k in range(n+1):  # go through excitations
        for it in range(len(v)):
            b = np.binary_repr(it, width=n) # binary representation (string)
            a = string_to_array(b)       # as an array
            # count excitations, 
            # if corresponding entry is negative, apply CZ:
            if np.sum(a) == k and v[it] == -1:
                # check which subsystems are excited
                subs = list(np.nonzero(a)[0])
                # add to gate list
                edge_set.append(subs)
                #apply CZ
                v = mdot([C_phase(n, [subs]), v])
    return edge_set
    
# for rand_hypergraph_state
def C_phase(n, edge_list):
    """ product of generalised C_phase on subsystems in edge_list, e.g. [[0,2,3], [4]]
        for n qubit system
        args: n : int, number of qubits
              subs : list of ints, subsystems
        returns: ndarray, gate
        
        Could also be done by putting the corresponding hypergraph state
        on the diagonal
    """
    N = 2**n
    # start with identity (not yet in diagonal form)
    
    D = np.ones(N)
    for subs in edge_list:
        l = len(subs)        
        C = np.ones(N)
        for it in range(n):
            b = np.binary_repr(it, width=n) # e.g.'0111'
            a = string_to_array(b)       # e.g. np.array([0,1,1,1])
            # whenever there are only '1's in subsystems, 
            # introduce a -1 in the Gate
            if np.sum(a[subs]) == l:
                C[it] = -1
        D = D*C
    return np.diag(D)


# manipulate hyper edges / part trace for hypergraphs
# based on shrink and delete mechanism.

def edge_shrink(edge_set, v0, renorm=False):
    """ shrink all edges containing vertex v0
        args:   edge_set : list of lists, e.g. [[0,1], [0,3,4,5], [1]]
                v0 : int, edge to be removed, e.g. 1
                renorm : bool, relabels edges if true (e.g. n -> n-1 etc., 
                such that vertice are always labeled 0,...,m
        returns: new edge_set, [[0], [0,3,4,5], []]
    """
    # copy list, so old list stays unmodified
    import copy 
    es_c = copy.deepcopy(edge_set)
    for edge in es_c:
        for vertex in edge:
            if vertex == v0:
                edge.remove(vertex)
                
    if renorm == True:
        es_c = renorm_edges(es_c, v0)
    return es_c

def edge_delete(edge_set, v0, renorm=False):
    """ delete all edges containing vertex v0 
        args:   edge_set : list of lists, ee.g. [[0,1], [0,3,4,5], [1]]
                v0 : int, edge to be removed, e.g. 1
                renorm : bool, relabels edges if true
        returns: new edge_set : e.g. [[[0,3,4,5]]
    """
    import copy 
    es_c = copy.deepcopy(edge_set)
    ed =  [edge for edge in es_c if v0 not in edge]
    
    if renorm == True:
        ed = renorm_edges(ed, v0)
    return ed

def renorm_edges(hh, v0):
    """after removing edge, relabels vertices
        args: h, list of lists, an edge_set
    """
    for it_edge in range(len(hh)):
        for it_vert in range(len(hh[it_edge])):
            if hh[it_edge][it_vert] >= v0:
                hh[it_edge][it_vert] = hh[it_edge][it_vert] - 1
    return hh


def ptrace_graph_state_edges_one_subs(h_l, v0, renorm=True):
    """ shrink and deletes graph by single vertex v0
        args:    h_list :  list of lists, specifying hypergraphs
                 v0     :  int, vertex to be deleted
        returns: h_out  :  list of lists, twice as long as input h_list
                          containing shrinked and deleted graphs
    """

    h_ret = list()
    for h in h_l:
        h_ret.append(edge_shrink(h, v0, renorm=renorm))
        h_ret.append(edge_delete(h, v0, renorm=renorm))
    return h_ret

def ptrace_graph_state_edges(hh, vv, renorm=True):
    """ returns list of edges for partially traced graph state,
        provided as edges.
        shrink and deletes graphs in hh by vertices contained in vv
        partial trace for single graph state: [h]
        args:    h_list :   list of edge sets, specifying (e.g. more than one)
                            hypergraph
                 vv     :   int, vertex to be deleted
        returns: hh_new :   list of lists, per vertex traced out generally
                            twice as long as input h_list
                            containing shrinked and deleted graphs
                            
    """
    assert type(hh[0][0]) == list #make sure its in the right format [h] for
    # a single graph state.
    vv = np.sort(vv) # needed for renormalisation of edges
    vv = vv[::-1]    # start tracing out from last subsystem

    import copy
    h_out = copy.deepcopy(hh)

    for v in vv:
        # trace out subsequently each subsystem
        h_out = ptrace_graph_state_edges_one_subs(h_out, v, renorm=renorm)
    return h_out

def ptrace_graph_state(n, h_list, vv):
    """ returns partially traced graph state, given list of edges.
        args: 
        h_list: list of edge sets, i.e. list of (possibly of a mixture of more than one) graph states
        n  :  int, current number of subsystems
        vv :  list of ints, subsystems to be traced out
        returns:
            ndarray, graph state
    """
    vv = np.sort(vv) # needed for renormalisation of edges
    vv = vv[::-1]    # start tracing out from last subsystem

    import copy
    hh = copy.deepcopy(h_list)
    hh_new = ptrace_graph_state_edges(hh, vv)
    
    # dimension of reduced system
    l = len(vv)
    new_n = n - len(vv)
    N = 2**new_n
    
    R = np.zeros([N,N])
    for h in hh_new:
        R = R + dm(graph_state_from_edge_set(new_n, h)) / 2**l
    return R


####  thermal states / ground states ####

def thermal(H):
    """ generate thermal/exponential/Gibbs state for given Hamiltonian H.
        thermal state exp(H) / tr exp(H)
        args:       H : ndarray
        returns:    ndarray, density matrix
    """
    t = sp.linalg.expm(H)
    return t / trace(t).real

def part_sum(H):
    """ return partition sum = tr exp(H)
    """
    return trace(sp.linalg.expm(H)).real

def Massieu(H):
    """ Massieu function log(Tr[exp(H)])  """
    return np.log(part_sum(H)).real


def spectrum(rho):
    """ spectrum for hermitian matrices """
    assert is_herm(rho)
    return sp.linalg.eigvalsh(rho)

def gap(H):
    """ returns spectral gap of Hamiltonian = difference of two smallest eigenvalues
        args:     ndarray, hermitian Operator aka Hamiltonian
        returns:  ndarray, eigenspace of smallest eigenvalue
    """
    evl_01 = sp.linalg.eigvalsh(H, eigvals = (0,1))
    gap = evl_01[1] - evl_01[0]
    return gap

def degeneracy(H, tol=1e-12):
    """ returns degeneracy of smallest eigenvalue in Hamiltonian
        args:     ndarray,  hermitian Operator aka Hamiltonian
        returns:  int, degeneracy of groundstate 

    """
    evl = sp.linalg.eigvalsh(H)
    E0 = evl[0]
    deg = sum( el < E0 + tol for el in evl )
    return deg

def groundstate(H, gap_tol=1e-12):
    """ returns the groundstate of the Hamiltonian (density matrix)
        !! if groundstate is degenerate, will only return eigenvector \
        to one of the smallest eigenvalues
        args:     ndarray, hermitian Operator aka Hamiltonian
        returns:  ndarray, eigenvector of smallest eigenvalue
    """
    evl, ev = sp.linalg.eigh(H, eigvals = (0,0))
    gs = ev[:,0]
    if gap(H) < gap_tol:
        print('ground state space degenerate.')
    return gs
    
def comm(A, B):
    """ commutator for hermitian matrices """
    return dot(A, B) - dot(B, A)
    
def acomm(A, B):
    """ anti-commutator for hermitian matrices """
    return dot(A, B) + dot(B, A)

def comm_s(P_s, Q_s):
    """ commutator for pauli strings, e.g. comm_s('XXI', 'ZIZ')
        returns 0 if [P, Q] = 0 and returns 1 if {P, Q} = 0.
        uses qecc module
    """
    qecc = try_import_qecc()
    return qecc.com( qecc.Pauli(P_s) , qecc.Pauli(Q_s) )

#### Multiple Systems ####

def identity(n, d=2):
    """ identity operator for n-qubits
        args: n, int : number of parties
              d, int, local dimension
    """
    return np.eye(d ** n)

def Id(n, d=2):
    """ alias for identity() """
    return identity(n, d=d)
    
def mmix(n, d=2):
    """ maximally mixed n-qubit state
        args:       n : int
        returns:    ndarray 
    """
    return np.eye(d ** n) / d ** n

def expect(rho, A, herm=True):
    """ expectation value, tr(\rho A)
        args: rho : array, density matrix
              s : string, observable in pauli basis, e.g. 'XIZII'
              herm : bool, true if operator is Hermitean 
                      (yields real expectation value)
        returns: float, expecation value
    """
    if herm == True:
        exp_val = trace(dot(rho, A)).real
    else:
        exp_val = trace(dot(rho, A))
    return exp_val
    

def expect_fast(rho, s, herm=True):
    """ fast expectation value for big Pauli matrices, given as string, eg. 'XZII'
        for qubits only

        traces out irrelevant subsystems before calculating expectation value,
        using the fact that Tr[M_AB . N_A \ot 1_B] = Tr[ Tr_B(M_AB) . N_A]
        (faster for observables with small support)
        args: rho : array, density matrix
              s : string, observable in pauli basis, e.g. 'XIZII'
              herm : bool, true if operator is Hermitean 
                      (yields real expectation value)
        returns: float, expecation value
    """
    n = len(s)
    if s == 'I'*n:
        return trace(rho)

    supp     = support_str(s) # support of A    
    s_red    =  shorten_to_non_triv_s(s) # remove identies 'I'

    #first trace over unnecessary subsystems, then take expecatian value
    trace_over = complement(supp, n)
    rho_red    = ptrace(rho, trace_over)
    
    #expectation value of Hermiean operators must be real...
    if herm == True:
        exp_val = trace( dot(rho_red, P(s_red) )).real
    else:
        exp_val = trace( dot(rho_red, P(s_red) ))
    return exp_val
    

#def expect_sparse(rho, s):
#    """ given sparse density matrix, calculates expectation value of Pauli-Observable.
#        e.g. s = 'XZYYIII'; use to_sparse() to obtain sparse form
#    """
#    n = len(s)
#    if s == 'I'*n:
#        return trace(rho)
#    s_red    =  shorten_to_non_triv_s(s) # removes identies 'I'
#    supp     = support_str(s) # support of A
#    subs     = np.arange(n) #subsystems
#    to       = [el for el in subs if el not in supp]
#
#    rho_sparse = to_sparse(ptrace(rho,to))
#    prod = dot(rho_sparse,P_sparse(s_red))
#    return trace( prod.todense() ).real     ### FIXME: trace operation in sparse

def ptrace_coeff(coeff, n, trace_over):
    """ takes partial trace over systems defined in trace_over over coefficients of Pauli basis
    args:   coeff : dict, Pauli coefficients
            n   : int, number of qubits
            trace_over : systems to be traced out, e.g. [2,3] or [3,0]
    """
    if len(trace_over) == 0:
        return coeff
    
    coeff_tr = dict()
    # work iteratively, start with last party to trace out
    # only keep terms with identity at that position
    to = trace_over[:] # make a copy of trace_over, otherwise it gets modified
    to = np.sort(to)

    l = to[-1] #consider last subsystem
    #throw away non identity elements
    coeff_tr = dict( (k[:l] + k[l+1:], v) for k, v in coeff.items() if k[l] == 'I' ) 

    if len(to) > 1:#more subsystems to trace out.
        n = n-1
        to = to[:-1]
        coeff_tr = ptrace_coeff(coeff_tr, n, to)
    return coeff_tr

# original ptrace
#def ptrace(rho, trace_over, d=2, padded=False):
#    """ partial trace over subsystems specified in trace_over for arbitrary 
#        n-quDit systems of equal dimensions
#        e.g. ptrace(rhoABC, [0]) = rhoBC        
#        if padded is True, state will be tensored back by identity to its
#        original size
#    args:       rho : ndarray
#                trace_over: list of subsystems to trace out, counting starts at 0
#                d : int, dimension (default is 2 for qubits)
#                padded : bool, pad by identity if True
#    returns:    rho_tr : ndarray
#    """
#    rho = make_dm(rho)
#    if len(trace_over) == 0: # if no subsystem gets trace out.
#        return rho
#    
#    n = number_of_qudits(rho, d)
#    trace_over = np.sort(trace_over)
#    rho2 = rho.reshape([d]*2*n) #two indices per local system
#    
#    idx = np.arange(2*n)
#    for i in trace_over:
#        idx[n+i] = i
#    
#    n_new = n - len(trace_over)
#    
#    # traced out state
#    rT = np.einsum(rho2, idx.tolist()).reshape(d**n_new,d**n_new)
#    
#    if padded == False:
#        return rT
#    else:
#        Sc = list(trace_over)
#        S = complement(Sc, n)
#        k = len(S)
#        return tensor_mix(rT, S, identity(n-k,d), Sc, d=d)


def ptrace(rho, trace_over, d=2, padded=False): # former ptrace_dim
    """ partial trace over subsystems specified in trace_over for arbitrary 
        n-quDit systems (also of heteregeneous dimensions)
        e.g. ptrace(rho_ABC, [1]) = rhoA_C
        if pad is True, state will be tensored by identity to its original size
    args:       rho :   ndarray
                trace_over: list of subsystems to trace out, counting starts at 0
                d :         int, dimension (default is 2 for qubits), or list of 
                            dimensions (for heterogeneous systems)
                pad :    bool, pad by identity if True
    returns:    rho_tr :    ndarray
    """
    rho = make_dm(rho)
    if len(trace_over) == 0: # if no subsystem gets trace out.
        return rho
    
    #create list of dimension [d1, d1, d2, d2, ...]
    if isinstance(d, int):
        n = number_of_qudits(rho, d)
        dims = [d] * n
        ddims = [d] * 2*n
    else:
        n = len(d)
        dims = d
        ddims = d+d
    
    trace_over = np.sort(trace_over)

    #reshaped matrix, two indices per local system
    rho2 = rho.reshape(ddims)
    
    idx = np.arange(2*n)
    for i in trace_over:
        idx[n+i] = i
    
    #calculate new number of particles and new dimensions
    syst_new = [i for i in range(n) if i not in trace_over]
    dims_new = [dims[i] for i in syst_new]
    dims_trace_over = [dims[i] for i in trace_over]
    d_new = to_int(np.product(dims_new))
    d_traced = np.product(dims_trace_over)
    
    # reshape traced out state
    rT = np.einsum(rho2, idx.tolist()).reshape(d_new, d_new)
    
    if padded == True:
        Sc = list(trace_over)
        S = complement(Sc, n)
        return tensor_mix(rT, S, np.eye(d_traced), Sc, d=dims_new + dims_trace_over)
    else:
        return rT


def ptrace_pauli(rho, trace_over):
    """ partial trace over subsystems specified in trace_over
        naive implementation: getting all pauli-coeffs, throwing away
        those having identity at position in trace_over (naive implementation)
    args:       rho : ndarray
                trace_over: list of subsystems to trace out, counting starts at 0
    returns:    rho_tr : ndarray
    """
    if len(trace_over) == 0: # if no subsystem gets trace out.
        return rho        

    n = number_of_qubits(rho)
    trace_over = np.sort(trace_over)

    b = len(trace_over)
    n_new = n - b
    kk = np.arange(n_new+1) # don't get all coeffs. only those up to weight n_new
    coeff = pauli_coeff(rho, kk)
    coeff_tr = ptrace_coeff(coeff, n, trace_over)
    return reconstruct_operator(coeff_tr, n_new)


def red_state_padded(rho, S, d=2):
    """ returns operator reduced onto S, padded by identity 
        wrapper for padded ptrace
    """
    n = number_of_qudits(rho, d)
    S = list(S)
    Sc = complement(S, n)
    return ptrace(rho, Sc, d=d, padded=True)


def ptranspose(rho, transpose_over, d=2):
    """ partial transpose over subsystems specified in transpose_over for arbitrary 
        n qu-dit systems
        e.g. ptranspose(rhoABC, [0]) = T_A (rho_ABC)
    args:       rho : ndarray
                transpose_over: list of subsystems to transpose, counting starts at 0
                d :  int, dimension (default is 2 for qubits), or list of 
                            dimensions (for heterogeneous systems)
    returns:    rho_tr : ndarray
    """
    if isinstance(d, int):
        n = number_of_qudits(rho, d)
        ds = [d]*n
        totaldim= d**n
    elif isinstance(d, list):
        n = len(d)
        ds = d
        totaldim= np.prod(ds)
    else:
        raise Exception("d should be an integer or a list of local dimensions!")
    transpose_over = np.sort(transpose_over)
    
    rho_R = np.reshape(rho, ds * 2) 

    idx = np.arange(2*n) # 2n indices
    for i in transpose_over: #exchange indices for partial transpose
        idx_i = idx[i]
        idx_ni = idx[n+i]
        idx[i]   = idx_ni
        idx[n+i] = idx_i
    
    rho_T = np.einsum(rho_R, idx.tolist())
    out   = np.reshape(rho_T, [totaldim, totaldim])
    return out


def reshuffle2(rho, d=2):
    """ reshuffles the two _outer_ indices of a bipartite system: 
        |ij><kl| -> |lj><ki|
    """
    rho_R = np.reshape(rho, [d] * 4)          # reshape into d x d x d x d
    rho_R = np.einsum("ijkl -> ljki", rho_R)  # reshuffle 
    rho_R = np.reshape(rho_R, [d**2, d**2])   # reshape into d^2 x d^2
    return rho_R


def reshuffle(rho, sys, d=2):
    """ reshuffling operation
        |...ij...><...kl...|  ->  |...lj...><...ki...|, where i and l live
        on subsystems defined by sys.
        sys : tuple, first index specifies ket, second index specifies bra
        
        Example: [0,1] reshuffles the outer indices i,l
                 [1,0] reshuffles the inner indices j,k
    #todo; generalise function to arbitrary index permuation.
    """
    
    n = number_of_qudits(rho, d)
    
    rho_R = np.reshape(rho, [d] * (2*n))
    t1 = sys[0]
    t2 = sys[1]
    idx = np.arange(2*n) # 2n indices
    
    idx_i  = idx[t1]
    idx_ni = idx[n+t2]
    idx[t1]   = idx_ni
    idx[n+t2] = idx_i
    
    rho_T = np.einsum(rho_R, idx.tolist())
    out   = np.reshape(rho_T, [d ** n, d ** n])
    return out


#### TENSORING ####
def mkron(systems):
    """ tensor product
        tensor product of elements in list, takes repeatingly the tensor product (sp.kron)
        with the last element in systems until all subsystems tensored up.
        natural (pen-paper) way to do it.
        #skip empty subsystems
    args:       systems : array_like, ndarray or list
    returns:    ndarray
    """
    
    tensor_product = 1
    for subsystem in systems[::-1]:
        tensor_product = sp.kron(subsystem, tensor_product)  # iteratively tensor subsystems
    return tensor_product

def mkronf(systems):
    """ fast tensor product
        fast tensor product of elements in list, using einsum
    args:       systems : array_like, ndarray or list
    returns:    ndarray
    """
    
    k = 0
    num = len(systems)
    dx = 1
    dy = 1
    params = []
    for s in systems:
        params.append(s)
        params.append([k, k+num])
        
        dx = dx * np.shape(s)[0]
        dy = dy * np.shape(s)[1]
        k+=1
        
    return np.einsum(*params).reshape(dx, dy)

def mkrons(systems):
    """ tensor product of sparse elements
        tensor product of sparse elements in list, takes repeatingly the tensor product (sp.kron)
        with the last element in systems until all subsystems tensored up.
        natural (pen-paper) way to do it.
        #skip empty subsystems
    args:       systems : array_like, ndarray or list
    returns:    ndarray
    """
    tensor_product = 1
    for subsystem in systems[::-1]:
        tensor_product = scipy.sparse.kron(subsystem, tensor_product)  # iteratively tensor subsystems
    return tensor_product


def mkron_self(syst, N):
    """returns N times self tensored system compare with mkron(systems)
    args:       N : int
    returns:    ndarray
    """
    tp_self = 1
    for i in range(N):
        tp_self = sp.kron(syst, tp_self)
    return tp_self


def direct_sum(a, b):
    """returns the direct sum of two matrices a and b
    args:       a, b : ndarray
    returns:    ndarray
    """
    dsum = np.zeros( np.add(a.shape,b.shape), dtype=np.complex )
    dsum[:a.shape[0],:a.shape[1]]=a
    dsum[a.shape[0]:,a.shape[1]:]=b
    return dsum


def tensor_mix(*args, **kwargs): # former tensor_mix_dim
    """Builds a tensored operator by splicing operators to given positions.
       For example, to build the operator Op1 x Op2, use tensormix(Op1, [0], Op2, [1]),
       to splice a two-party operator Op1 to subsystems 1 and 3 and a single-party
       operator to subsystem 2, use tensormix(Op1, [0,2], Op2, [1]).
    args:       op1 places1 [op2 places2, op3 places3, ...] [d = list of local dims]
    returns:    ndarray
    """

    # if no dimensional parameter is given, assume that every system is of dimension 2
    d = 2
    if 'd' in kwargs:
        d = kwargs['d']
    
    # number of systems to mix
    n_mix = len(args)//2
    
    # convert dimension into list of dimensions if necessary
    n_sys = 0
    for i in range(n_mix):
        n_sys += len(args[2*i+1])
    
    if not isinstance(d, list):
        d = [d]*n_sys
    
    offset = 0
    
    n    = []
    rhop = []   # list of reshaped operators
    sys  = []    # list of its target places
    name = []   # list of names needed for einsum
    
    # for every operator to mix...
    for i in range(n_mix):
        rho1 = args[2*i]    # operator
        sys1 = args[2*i+1]  # its position
        n1 = len(sys1)
        
        #...assign a unique name (needed to address it by einsum)
        name1 = np.arange(2*n1)+offset
        #...reshape it properly. For example, P('XZ') comes as a single 4 by 4 matrix, but we need it as a 2x2x2x2 array
        rhop.append(rho1.reshape([d[sys1[i]] for i in range(n1)]*2))
        #...save size, target positions and name to list
        n.append(n1)
        sys.append(sys1)
        name.append(name1)
        
        offset += 2*n1
    
    num_systems = sum(map(lambda x: len(x), sys))
    
    #build target index relabeling for einsum
    idx1 = [-1]*num_systems
    idx2 = [-1]*num_systems
    
    for k in range(n_mix):
        for s in range(len(sys[k])):
            idx1[sys[k][s]] = name[k][0]+s
            idx2[sys[k][s]] = name[k][n[k]]+s

    #Circumvent Bug in einsum, not accepintg input when type is int64: Workaround is to convert them to python int
    idx1 = list(map(int, idx1))
    idx2 = list(map(int, idx2))
    
    for k in range(len(name)):
        name[k] = list(map(int, name[k]))
    
    einargs = list(sum(zip(rhop, name), ())) + [idx1+idx2]
    
    size = 1
    for dd in d:
        size = size*dd
    
    return np.einsum(*einargs).reshape(size,size)





#### SECTOR RELATIONS IN THE BLOCH DECOMPOSITION. ####
    
def sector(rho, k):
    """ returns sector P_k,
    sum of all pauli terms with weigth k  in operator """
    n = number_of_qubits(rho)
    c = pauli_coeff(rho, [k])
    return reconstruct_operator(c, n) * 2**n

### qubits
def sector_len(r, k_vec=None):
    """ returns sum of squares of bloch coefficients
        corresponding to pauli-terms of weight k
        \sum c_i^2  where |A_i| = k
    """
    if k_vec == None:
        n = number_of_qubits(r)
        k_vec = np.arange(n+1)
        
    if type(k_vec) == int: #if only a number is provided
        k_vec = [k_vec]

    l = len(k_vec)
    S = np.array(np.zeros(l))
    r = make_dm(r)
    
    for it in range(l):
        k = k_vec[it]        
        c = pauli_coeff(r, k)
        S[it] = np.sum([np.abs(value)**2 for value in c.values()])
    return S

def sector_len_f(r, d=2):
    """ obtains sector lenghts / weights trough purities and Rains transform 
        faster, and for arbitrary dimensions
    """
    n = number_of_qudits(r, d=d)
    Ap = np.zeros(n+1)

    for k in range(n+1):
        for red in all_kRDMs(r, k, d=d):
            Ap[k] = Ap[k] + purity(red)
    # transform (Rains) unitary to Shor-Laflamme primary enumerator
    import qcode_enum as qce
    Wp = qce.make_enum(Ap, n)
    W = qce.unitary_to_primary(Wp, q=d)
    #A = qce.poly_coeffs(W)
    A = qce.W_coeffs(W)
    u = np.array([float(el) for el in A])
    # pad with zeros to length n+1
    if len(u) < n+1:
        u = np.hstack([u, np.zeros(n+1 - len(u))])
    return u


def comb_sr(n,k,j):
    """ combinatorial term for sector formula 
        binom(n,k) * binom(k,j) / binom(n,j)
        = binom(n-j,n-k)                        """
    if j == 0: return int_binom(n,k)
    else:    return int_binom(n-j,n-k)

def comb_sr2(n,m,k,j):
    """ combinatorial term for monogamy sector formula """
    return int_binom(m,k) * int_binom(k,j) / int_binom(n,j) # rausgenommen: * binom(n,m)
    
    
### n odd, m in {0,..., n/2}

#cut relations (from Schmidt-decomposition)
def sector_relations_cut_odd(n, q, m, verbose=True):
    """ prints sector relation equations obtained from equal purity across a cut
        for n odd, m in {0,..., n/2}
    """    
    assert is_odd(n) and m>=0
    a = n // 2
    
    rhs = ( q**(2*m+1) -1) * int_binom(n, a-m)
    if verbose:
        print( rhs , ' = ', end=' ')
    
    Coeffs = np.zeros(n) # vector of coefficients
    for j in range(1,a-m+1): #lower sectors
        factor= -(q**(2*m+1) - int_binom(a+1+m, j) / int_binom(a-m, j)) * comb_sr(n, a-m, j)
        Coeffs[j-1] = factor
        if verbose:
            print( print_sign(factor), abs(factor), '*A' + str(j) + '', end=' ')

    for j in range(a-m+1,a+m+2): #higher sectors
        factor = comb_sr(n, a+1+m, j)
        Coeffs[j-1] = factor
        if verbose:
            print( print_sign(factor), abs(factor), '*A' + str(j) + '', end=' ')
    
    if verbose: 
        print(';\n')
    return Coeffs, rhs

### n even, m in {1,..., n/2}
def sector_relations_cut_even(n, d, m, verbose=True):
    """ prints sector relation equations obtained from equal purity across a cut
        for n even, m in {1,..., n/2}
    """
    assert is_even(n) and m >=1
    a = n // 2
    
    rhs = np.zeros(1)
    rhs = rhs + (d**(2*m) -1) * int_binom(n, a-m)
    if verbose: 
        print( rhs , ' = ', end=' ')

    Coeffs = np.zeros(n) # vector of coefficients #, dtype=np.int
    for j in range(1,a-m+1): #lower sectors
        factor= -(d**(2*m) - int_binom(a+m, j) / int_binom(a-m, j)) * comb_sr(n, a-m, j)
        Coeffs[j-1] = factor
        if verbose:
            print( print_sign(factor), abs(factor), '*A' + str(j) + '', end=' ')

    for j in range(a-m+1,a+m+1): #higher sectors
        factor = comb_sr(n, a+m, j)
        Coeffs[j-1] = factor
        if verbose:
            print( print_sign(factor), abs(factor), '*A' + str(j) + '', end=' ')
            
    if verbose: print('\n')
    return Coeffs, rhs

def sector_relations_cut(n, q, verbose=True):
    """ return all Bloch sector equations relations as matrix
        args: n : int, number of parties
              q : int, local dimension
        returns:
              A, b, such that Av = b, where v is the vector composed 
              of v = (S1, S2, ..., Sn)
    """
    nhf = n // 2

    if is_odd(n):
        b = np.zeros([nhf+1])   # dtype=np.int
        A = np.zeros([nhf+1,n]) # dtype=np.int
        for m in range(nhf+1):
            A[m,:], b[m] = sector_relations_cut_odd(n, q, m, verbose=verbose)

    elif is_even(n):
        b = np.zeros([nhf]) #, dtype=np.int
        A = np.zeros([nhf,n]) # dtype=np.int
        for m in range(1,nhf+1): # m starts at 1..
            A[m-1,:], b[m-1] = sector_relations_cut_even(n, q, m, verbose=verbose)
    return A, b
    
    
### projector sector relations for k-uniform state
def sector_relations_projector(n,d,k):
    """ projector relations for k-uniform pure states
    
        weight enumerator has to fulfill linear equations 
        obtained from Tr [rho_(j)^2] = d^(2j-n)
        v = (S1, S2, ..., Sn)
        Av = b
        args:   n : int, number of parties
                d : int, local dimension
                k : int, k-uniformity of state
    """
    ne = k #number of equations
    A = np.zeros([ne, n]) # dtype=np.int
    b = np.zeros([ne])     # dtype=np.int
    
    # calculate a row corresponding to linear equation,
    # obtained from Tr [rho_(h)^2] = d^(2h-n)
    
    for it in np.arange(0,k): # n-k <= m < n
        m = (n-k) + it # k': size of reduced system under consideration, bigger than n-k.
        b[it]   = ( d**(2*m-n) - 1 ) * int_binom(n,m)
        
        for j in range(1, m+1):     # 1 <= j <= m
            A[it,j-1] =  comb_sr(n,m,j) # * S[j-1]
    
    return A, b

def sector_relations_additive(n, codetype='odd'):
    """ sector length for additive codes / graph states:
        sum of even sector length is either
        a)   2^(n-1) - 1  (odd / type I)
        b)   2^n - 1      (even / type II) 
        proof: rho^2 = rho, apply parity lemma
    """
    A = np.zeros(n) #, , dtype=np.int
    A[1::2] = 1 # A = np.array([0,1,0,1,0,...])
    if codetype == 'odd':    # type I
        b = 2**(n-1) - 1
    elif codetype =='even': # type II
        b = 2**n - 1
    return A, b

def sector_relations_mono2(n,q):
    """ only single full state sector relation from 2nd degree monogamy.
        multiplied by a factor q**n!
        included in sector_relations_mono2_red below
        d : int, local dimension
        # this is the zero'th shadow coefficient S0.
    """
    A = np.zeros(n+1) # A0,...,An
    for a in range(0, n+1):
        for j in range(0,a+1):
            A[j] = A[j] + comb_sr(n,a,j) * (-1)**a * q**(n-a) # multiplied by q**n.
    b = -A[0] # split off identity term = A0    
    A_out = A[1:]
    return A_out, b


def sector_relations_mono2_red(n,q,k=0):
    """ monogamy relation of second degree for all reduced subsystems
        require Ax >=b for weight distribution x
        for even & odd n, all dimensions. Follows from universal state inversion \Lambda(rho)
        \tr [ \Lambda(rho) rho ] >= 0.
        args:   n : int, number of parties
                k : int, k-uniformity of state. (skips first k lines of matrix)
                d : int, local dimension
    """
    A = np.zeros([n-k, n+1]) # A0,...,An
    for m in range(k+1,n+1): # reduced systems of size  k+1 <=m <= n
        for a in range(0, m+1):
            for j in range(0,a+1):
                A[m-k-1,j] = A[m-k-1,j] + comb_sr2(n,m,a,j) * (-1)**a * q**(n-a)
    b = -A[:,0] # split off identity term = A0
    A_out = A[:,1:]
    return A_out, b
    
def sector_relations_mono2_red_f(n,q,k=0): #correct?
    """ monogamy relation of second degree for all reduced subsystems
        faster formula
        require Ax >=b for weight distribution x
        for even & odd n, all dimensions. Follows from universal state inversion \Lambda(rho)
        \tr [ \Lambda(rho) rho ] >= 0.
        multiplied by factor q**n!!
        args:   n : int, number of parties
                k : int, k-uniformity of state. (skips first k lines of matrix)
                d : int, local dimension
    """
    A = np.zeros([n+1, n+1]) # A0,...,An
    for m in range(k+1, n+1): # reduced systems of size  k+1 <=m <= n
        for j in range(0,m+1):
            A[m,j] = (q-1)**(n-j) * (-1)**j  * comb_sr(n,m,j)
            
    b     = -A[k+1:,0] # split off identity term = A0    
    A_out = A[k+1:,1:]
    return A_out, b


def sector_relations_eq(n,d,k=0,additive=False, codetype='odd'):
    # equality
    Ac, bc = sector_relations_cut(n,d, verbose=False)
    Ap, bp = sector_relations_projector(n,d,k)

    A = np.vstack([Ac, Ap])
    b = np.hstack([bc, bp])

    if additive == True:
        Aa, ba = sector_relations_additive(n,codetype)
        A = np.vstack([A, Aa])
        b = np.hstack([b, ba])
    return A, b


def sector_sdp(A, b, A_ineq=None, b_ineq=None, solver='mosek', 
               verbose=True, vtype='continuous', tol = 1e-6, \
               mosek_params={'infeas_report_auto': 0} ):
    """    A1*x == b1, A2*x >= b2 """
    
    import picos as pic
    import cvxopt as cvx
    ## initialize Problem
    sdp = pic.Problem()

    #### equality constraint
    M,N = np.shape(A)
    ## convert numpy arrays to cvx matrices:
    Am   =   cvx.sparse(cvx.matrix( A, (M,N)) )
    bm   =   cvx.sparse(cvx.matrix( b, (M,1)) )

    ## add matrices as parameters:
    A1  = pic.new_param('A1',  Am )
    b1  = pic.new_param('b1',  bm )

    #### inequality constraint
    if A_ineq is not None:
        print('include inequalities...')
        MM,NN = np.shape(A_ineq)         #sizes
        
        ## convert numpy arrays to cvx matrices:
        A_ineqm   =   cvx.sparse(cvx.matrix( A_ineq, (MM,NN)) ) #for size 1xn arrays, replace NN by 1
        b_ineqm   =   cvx.sparse(cvx.matrix( b_ineq, (MM,1 )) )
        ## add matrices as parameters:
        A2  = pic.new_param('A2',  A_ineqm )
        b2  = pic.new_param('b2',  b_ineqm)
        

    ## define Variable to be optimised:
    x = sdp.add_variable('x', N, vtype=vtype)

    ## add constraints:
    sdp.add_constraint( x > 0)
    #sdp.add_constraint( x < d**n) # unnecessary, as sector sum & positivity imply this.

    #if A is not None:    
    sdp.add_constraint( A1*x == b1 )
    if A_ineq is not None:
            sdp.add_constraint( A2*x >= b2 ) # use | if A is just a row..

    ## set objective:
    #sdp.set_objective('find', x)
    # 'find' is currently broken in Picos, see https://gitlab.com/picos-api/picos/issues/124
    #replacement
    sdp.set_objective('max', 1|x)

    ## solver settings
    solver       = solver # solver = 'mosek7', 'cvxopt'
    solv_tol     = tol    # seems to be lowest tolerance that doesn't return most problems as UNKNOWN 

    if verbose: print(sdp)
    ## solve
    sol = sdp.solve(solver=solver, tol=solv_tol, verbose=verbose, mosek_params=mosek_params) #, maxit=max_it
    #status = sol['status']
    
    x0 = None
    try:     
        x0  = np.array(x.value)
    except:  
        x0 = 'no solution or certificate returned'
    return sdp, sol, x0


#alternative to sector_sdp:
#from scipy.optimize import linprog
#c      =  -np.ones(n)
#linprog(c, A_ub=None, b_ub=None, A_eq=A, b_eq=b, bounds=(0, None), \
#        method='simplex', callback=None, options=None)


#### ACTIONS ####

def conj_act(A, g):
    """ conjugated action of g on A"""
    return mdot([g, A, dag(g)])

#### REDUCED DENSITY MATRICES ####


def all_kRDMs(rho, n, k=2, d=2, padded=False, verbose=False):
    """ generator of all reduced states of size k, 
        optionally padded with identities 
    """
    l = numpy.arange(n)
    
    # to:     trace over subs
    # to_bar: subs in complement (what remains after partial tracing)
    for to in itt.combinations(l, n-k):
        to = list(to)
        if verbose:
            print ('trace over', to)
        rho_red = ptrace(rho, to, d=d)
        to_bar = list(set(l) - set(to)) #complement

        
        if padded == False: ## padded (tensored) with Identity
            yield rho_red
        else:
            yield tensor_mix(rho_red, to_bar, identity(n-k,d), to, d=d)

def all_purities(rho, d=2):
    """ returns Rains unitary enumerator, given operator rho """
    n = number_of_qudits(rho, d)
    A = np.zeros(n+1)
    for k in range(n+1):
        for el in all_kRDMs(rho, k=k, d=d):
            A[k] = A[k] + purity(el)
    return A
            
#### UNIVERSAL STATE INVERSION & RELATED ####

def invert_state(r, d=2, c=1):
    """ generalised state inversion """
    n = number_of_qudits(r,d)
    #initialise
    A = np.zeros([d**n, d**n])

    for k in range(0,n+1):
        for Marginal in all_kRDMs(r, k, d=d, padded=True):
            A = A + (-1/c)**k * Marginal
    return A
    
def gen_invert_state_T(rho, T, d=2, C_ST=1, C_STc=1):
    """ implementing Rains state inversion formula, and its generalisation """
    n = number_of_qudits(rho, d)
    systems = np.arange(n)
    m = len(set(T))
    assert m <= n
    
    A = np.zeros([d ** n, d ** n]) #initialise
    
    for k in range(0,n+1): # k = |S|
        for S in itt.combinations(systems, k):
            
            S = list(S)               # k = |S| #Sc = complement(S, systems)
            Tc = complement(T, systems)
            a  = len(set(T)  & set(S))    # overlap between S and T
            ap = len(set(Tc) & set(S))    # overlap between S and Tc
            assert a <= k; assert a <=m
            
            #print('S, Sc', S, Sc) #print('T, Tc', T, Tc) #print('a, ap', a, ap)
            rho_S = red_state_padded(rho, S, d=d)
            
            A = A + (-1 / C_ST)**a  *  ( 1 / C_STc)**ap * rho_S
    return A

def gen_invert_state_sym(rho, m, d=2, c=1):
    """ symmetrised inverted state with |T|=m """
    import qcode as qc
    n = number_of_qudits(rho, d)
    assert m <= n
    
    A = np.zeros([d ** n, d ** n]) #initialise
    
    for k in range(0,n+1):
        kMarginal_sum = np.zeros([d ** n, d ** n])
        for Marginal in all_kRDMs(rho, k, d=d, padded=True):
            kMarginal_sum = kMarginal_sum + Marginal
        A = A +   qc.Krawtchouk(m,k,n,c) * kMarginal_sum * (-1)**k
    return A / c**m


def channel(rho, K_list, d=2):
    """ applies channel to rho given list of Kraus operators K_list """
    assert type(K_list) == list
    n = number_of_qudits(rho, d)
    out = np.zeros([d ** n, d ** n])
    for K in K_list:
        out = out + mdot([K, rho, dag(K)])
    return out


#### k-UNIFORMITY TESTS ####s

def qecc_k_uni_check_from_adj(adj):
    """ Given adjacency matrix of graph state,
        Checks if all k-RDMs are maximally mixed using the qecc library"""
    import time; tic = time.time()
    gen    = adj_to_generator(adj)
    wt_min = min_weight_stabiliser(gen)
    print('min_weight_stabiliser:', wt_min)
    print_elapsed(tic)
    return wt_min

def fast_k_uni_check_from_adj(k, adj, verbose=True):
    """ Given adjacency matrix of graph state,
        Checks if all k-RDMs are maximally mixed.
        Short circuits if single weight below k found.
        args:   
            n : int, number of qubits
            adj: nxn adjacency matrix
            k : int, up to which level k-RDMs should be tested
            verbose : bool, for verbose output
        returns:    
            wt_min      : int, smallest weight of any stabiliser element found
                          or if below k, weight of any stabiliser element smaller than k
            choice_min : list of ints, RDM corresponding to wt_min (not max mixed k-RDM)
    """
    #import time; tic = time.time()
    n = np.dim(adj)
    wt_min = np.inf
    subs = np.arange(n) # all subsystems
    choice_min = 'None'
    
    # check that generators have minimum weight:
    wt_gen = np.min(np.sum(adj, 0)) + 1
    if wt_gen <= k: # add 1 to account for 'X' at vertex
        return wt_gen, 'None'
 
    for number_of_generators in range(2,k+1): # check up to k generator products
        g = itt.combinations(subs, number_of_generators) #all possible k-RDM subsystems
#        if verbose: print('multiplying', number_of_generators, 'generators')

        while True:
            try:     choice = g.next() # find new partition
            except:  break # if no more combinations left                
        #for it in range(n_trials):  #choice = np.random.choice(subs, size=(n-k), replace=False)
                
            ### multiply all generators modulo 2 == summing
            #col_sum = np.sum(adj[choice, :], axis=0)      # picking out rows of choice, summing over columns
            #col_sum_trim = np.delete(col_sum, choice)     # delete columns where there are X's
            #col_sum_trim_mod2 = np.mod(col_sum_trim, 2)   # modulo 2 to account for Z^2
            #col_sum_trim_mod2_row_sum = np.sum(col_sum_trim_mod2)  # sum remaining elements
            # concatenated:
            col_sum_trim_mod2_row_sum = np.sum(np.mod(np.delete(np.sum(adj[choice, :], axis=0), choice), 2)) 
            wt = to_int(col_sum_trim_mod2_row_sum + len(choice)) # add back X's
            
            #if any weight below k found, keep track of it, and short-circuit.
            if wt <= k: #abort test if k-RDM found which is not max mixed. 
                if verbose == True: print('weight below/equal k found', wt, 'example generator:', choice)
                return wt, choice
    
                #if wt <=wt_min: choice_min = choice; wt_min = wt   #uncomment if truly lowest weigth has to be found!!
#        if verbose: print_elapsed(tic)
#            
    if verbose:
        print('number of qubits', n)
        print('weight (above/below/equal) k found', wt_min)
        print('example generator:', choice_min)
#        print_elapsed(tic)
    return wt_min, choice_min



def find_k_uniform_states(n, k, edge_degree = None):
    """ finds a subset of k-uniform graph states whose adjacency matrix has 
        off diagonal band structure / symmetric. (toeplitz could be an extra condition)
        some of these matrices are circulant (toeplitz + symmetric)
        args:
            n : int, number of qubits
            k : int, k-uniformity
        returns: 
            list of adjacency matrices characterised by nonzeros in offdiags
    """
    tic = time.time()
   
    offdiag_list = list()
    subs = np.arange(n)

    if edge_degree == None:
        edge_degree = k    
    # adjacency matrix needs to have a 0 in diagonal, and k 1's in offdiagonal
    g = itt.combinations(subs[1:], edge_degree)
    
    while True:
        try: # get new cyclosymmetric graph.
            graph_choice = g.next() # get a new list of offdiags. e.g. (1, 3, 4)
            
            # impose symmetry of matrix: symmetric circulant. too strong condition!
            #gc = np.asarray(graph_choice)            
            #if not np.product( gc == n - gc[::-1]): continue
        except:
            break
        #print update
        #it +=1; sys.stdout.write('\r' + str(it) + '/' + str(binom(len(subs)-1, k))), sys.stdout.flush()
        #choice = np.random.choice(a[1:], size=(n-1-k), replace=False)  # statistic testing
        adj = circulant_matrix(n, graph_choice)
        wt, choice_min = fast_k_uni_check_from_adj(k, adj, verbose=False)

        if wt >= k+1: # keep graphs which are k-uniform
            offdiag_list.append(graph_choice)  # print(graph_choice)
            print('\r', 'n =', n, ';  k=', k, ';  offdiags', offdiag_list)
            
    print('\n'), print_elapsed(tic)
    #print('number of cyclosymmetric graphs', int(binom(n-1,k)))
    print('\r', 'n=', n, ';  k=', k, ';  offdiags', offdiag_list)
    print('all choices tested.')    
    return


def k_uni_check_ptrace_from_edges(n, k, edg, rand=True, m = 13):
    """ statistic or complete testing of k-uniformity for graph states, given edges.
        speed: medium
    """
    import qdict as qdc
    print(int_binom(n, k), 'k-RDMs in total')
    a = np.arange(0,n)
    
    # either random k-RDMs tests, or systematic all partitions
    if rand == True:
        print('testing', m, 'random k-RDMs...')
        for it in range(m):
            choice = np.random.choice(a, size=(n-k), replace=False)
            red = ptrace_graph_state(n, [edg], choice)
            #compare to identity
            assert np.linalg.norm(red - mmix(k)) <= 1e-14, print (choice, qdc.dict_clean(pauli_coeff(red)))
    else:
        g = itt.combinations(a,(n-k))
        print('testing all k-RDMs.. ')
        while True:
            choice = g.next()
            red = ptrace_graph_state(n, [edg],choice)
            #compare to identity
            assert np.linalg.norm(red - mmix(k)) <= 1e-14, print(choice, qdc.dict_clean(pauli_coeff(red)))
    print('if no assertion error displayed, all k-RDMs tested maximally mixed')
    return


#### INFORMATION GEOMETRY ####

def inf_projection(rho, k, it_max = 2000, delta=0.7, verbose=True, sym=None, real=False):
    """ Calculate information Projection rho_k
        Obtain MaxEnt state having the k-RDMs (as specified in basis B) of M 
        J. Phys. A:Math. Theor. 46 (2013) 125301 (16pp), 
        S. Niekamp et al. - Computing complexity measures for quantum states based on exponential families
        Arg: 
            rho    :  ndarray, states whose information projection is to be calculated
            k      :  int, specidying exponential family Qk, i.e. k-marginals of rho and rho_k have to match
            it_max :  int, number of approximation iterations
            delta  :  float, underrelaxation parameter
            verbose:  bool, verbositey
            sym    :  list containing ints, specifying permutation symmetries 
                      of subsystems e.g. [0,2]
            real   :  bool, true if density matrix has real entries only
        returns: 
            tau    :  ndarray, information projection rho_k (here: tau) of rho
            Hk     :  ndarray, associated Hamiltonian
            dev    :  float, deviation from ideal expectation values
            t      :  float, time to solution in seconds
    """
    import time; tic = time.time()
    n = number_of_qubits(rho)
    N = 2**n #dimension of multi-qubit system
    
    kk = np.arange(1, k+1)
    B_red = pauli_basis_k_local_str(n, kk) # reduced k-local basis as list of strings

    if real == True:
        B = [el for el in B_red if el.count('Y') % 2 == 0]
    else:
        B = B_red
    l = len(B)
        
    # precompute expectation values of k-local observables of rho
    rho_exp = [expect(rho, P(el)) for el in B]
    
    # updates Hamiltonian
    Hk = np.eye(N)                  # starting Hamiltonian    
    for it in range(it_max):
        
        r      =  rnd.randint(0,l-1) # 0 <= r <= l (pick observable)
        A      =  B[r]
        A_rho  =  rho_exp[r]        # precomputed expectation value
        #A_tau  =  expect(thermal(Hk), A)
        A_tau  =  expect_fast(thermal(Hk), A)  # faster for big n and small support of observables

        #calculate eps
        try: # catching division by zero..
            a = 1 / (1 - A_tau**2)
        except ZeroDivisionError:
            continue
        eps    =  ( A_rho - A_tau ) / a
                
        # taking into account symmetry
        if sym is None:
            Hk     =  Hk + delta*eps* P( B[r] )
        else:
            Hk     =  Hk + delta*eps* P_sym( B[r], sym )
    
    # compare k-local coefficients
    tau    = thermal(Hk)               # final information projection
    c_rho  = pauli_coeff(rho, k = kk)
    c_tau  = pauli_coeff(tau, k = kk)
    dev = 0    
    # sum differences in expectation values
    for key, value in c_rho.items():   
        dev = dev + abs(c_rho[key] - c_tau[key])

    dt = int(time.time() - tic)
    if verbose == True:
        print_elapsed(tic)
        print('total deviation from desired values:', dev)
    return tau, Hk, dev, dt



### probabilistic functions
    
def LU_equivalence(p1, p2, dims = 2, it_max = 1, opt_over=None, repeat=1):
    """ tests LU equivalence by random unitaries.
        'it_max' tries per site tries, optimized over parties in 'opt_over';
        repeated 'repeat' times
        dd : list of ints, local dimensions
        returns overlap of pure states / transition probability
        
    """
    import time; tic = time.time()
    from qop import rand_unitary
    n = len(dims)
    ovmax = 0
    Umax = 0
    U_list = [np.eye( dims[it] ) for it in range(n)]

    if isinstance(dims, int):
        n = number_of_qudits(p1, d=dims)
        dims = [dims]*n
        
    if opt_over is None:
        opt_over = np.arange(n)
        
    for rep in range(repeat):
        print('round', rep)
        for sub in opt_over:
            for it in range(it_max):
                U_list[sub] = rand_unitary(dims[sub])
                U = mkron(U_list)
                ov = abs(dot(p1, dot(U,p2)))
                if ov > ovmax:
                    Umax  = U_list
                    ovmax = ov
            print_elapsed(tic); print(ovmax)
    
    return ovmax, Umax


#### special matrices ####

def eye_nonsquare(m,n):
    """ non-square mxn eye-matrix / identity matrix """
    Z = np.zeros([m,n]);
    np.fill_diagonal(Z, 1)
    return Z

def off_diagonal(d, u):
    """ d-dim matrix with offdiagonal at the u'th offdiagonal """
    return np.diagflat(np.ones(d-abs(u)), u)

def circulant_matrix(d, u_list):
    """ returns symmetric circulant matrix 
        having 1s in +/-! offdiagonals specified in u_list
        e.g. circulant_matrix(4,[1,3]) """
    circ = np.zeros([d,d])
    for u in u_list:
        if u == 0:
            circ = circ + np.eye(d)
        else:
            circ = circ + off_diagonal(d,u) + off_diagonal(d, -u) #below and above!!!
    return circ
    
def is_circulant(n, u):
    """ checks if first row of matrix represents a circulant matrix """
    u = np.array(u)
    return bool( np.product(u == n - u[::-1]) )


#### GEOMETRY ####

def hypercube_adj(n):
    """ creates adjacency matrix of hypercube, 
        according to the recursive formula
        A_k = [[A_{k-1}, Id], [Id, A_{k-1}]]    
    """    
    if n == 1:
        return np.array([[0,1],[1,0]])
    else:
        n_ = 2**(n-1)
        return np.vstack( [ np.hstack([hypercube_adj(n-1), np.eye(n_)]), \
                            np.hstack([np.eye(n_), hypercube_adj(n-1)]) ] )     
    
def vol_sphere(N):
    """ volume of hypersphere in N dimensions """
    return mt.pi**(N/2) / mt.gamma( N/2 + 1 )
    
def area_sphere(N):
    """ area for sphere S^N. i.e. S^2 is the surface of the regular sphere """
    return vol_sphere(N) * (N+1)
    

#### VISUALISATION ####

def draw_graph_adj(adjacency_matrix, layout='circo', spec=False):
    """ given an adjacency matrix use networkx and matlpotlib to plot the graph """
    import networkx as nx; import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    G=nx.Graph(adjacency_matrix)
    pos=graphviz_layout(G, prog=layout)    #   sfdp, twopi, neato, circo
    if spec == True: # draw Laplacian spectrum
        nx.draw_spectral(G, node_color='w',edge_color='b',width=3,edge_cmap=plt.cm.Blues,with_labels=True)
    else:
        nx.draw(G, pos, with_labels=True)
    plt.show()
    # without labels: nx.draw_networkx(G,with_labels=False)
    return

#### OUTPUT ####

def print_sign(number):
    if number >= 0: 
        return ' + '
    else: 
        return ' - '

def print_elapsed(tic):
    """ returns elapsed time, given starting time """
    import time    
    print(int(time.time() - tic), 'sec /', int((time.time() - tic)/60), 'min')
    return

def pprint_matrix(data, digits=3):
    """Print a numpy array as a latex matrix."""
    header = (r"\begin{{equation}}\begin{{pmatrix}}"
              r"{d}\end{{pmatrix}}\end{{equation}}")
    d=data.__str__()[1:-1]
    d = d.replace(']', '')
    d = d.replace('\n', r'\\')
    d = re.sub(r' *\[ *', '', d)
    d = re.sub(r' +', ' & ', d)
    display.display_latex(display.Latex(header.format(d=d)))

#### VARIOUS ####

def is_bool(c):
    return type(c)==bool

def is_even(k):
    return int(k % 2) == 0
def is_odd(k):
    return int(k % 2) == 1        

def delta(i,j):
    return i == j

def to_int(c):
    """ returns closest integer. introduces rounding errors. """
    return np.int(np.rint(c)) #np.int(np.round(c))

def int_binom(n, k):
    """ binomial coefficient as integer (not float!) 
        returns: int, binom(n,k)
        binom(n,k) = n! (n-1) ... (n-k+1)! / k! (k-1)! ... 1! 
    """
    if not n >= k >= 0:
        return 0
    prod = Fraction(1)
    for it in range(n-k):
        prod = prod * Fraction(n-it, n-k-it)
    assert prod.denominator == 1
    return prod.numerator

def dec2number(x, n, number):
    """ turns base 10 x into base number x, filling on the left until width is 
        n digits
    """
    s = ''
    if x == 0:
        return str.zfill(s, n)   # pad on the left with zeros
    while x > 0:
        s = str(x %  number) + s         # modulo
        x = x // number                  # integer division
    return str.zfill(s, n)

#### LIST & SET TOOLS ####

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itt.chain.from_iterable(itt.combinations(s, r) for r in range(len(s)+1))

def subsets(array):
    return [list(el) for el in powerset(array)]        


def complement(S, V):
    """ complement of S in in V (or alternatively of n parties)
        e.g. complement(np.arange(n), [1,4,5])
    """
    if type(V) == int:
        V = np.arange(V)
    return list(set(V) - set(S))

def sort_unique(sequence):
    """ unique sort for non-hashable type of elements 
        e.g. list of lists
        returns: generator
    """
    return (x[0] for x in itt.groupby(sorted(sequence)))

def list_rotate(l,x):
    """ shifts list cyclic to the right """
    return l[x:] + l[:x]
def cycle_list(l,x):
    return list_rotate(l,x)
    

#### STRING TOOLS ####

def string_to_array(s):
    """ given numbers in string, returns array """
    l = list(s)
    n = np.array(map(int, l))
    return n
    
def array_to_string(arr):
    """ given array of integers, maps it to a str mainly used for ket functions"""
    l = list(arr)
    l_int = map(str, map(int, l)) # first map to int, then to strings
    return ''.join(l_int)

def string_permutations_unique(s):
    """ returns unique permutations of elements in string """ #TODO is this really unique?
    s_perm = set(itt.permutations(s)) # create set of tuples (that's how itertools works)
    return [''.join(el) for el in s_perm]

def find_letter_in_string(s, letter):
    """ finds all occurrences of a letter in string """
    for pos, substring in enumerate(s):
        if substring == letter:
            yield pos
            
def string_shift(s, k):
    """ shift string by k letters, putting the [0:k-1] letters of string to the end
        shifts to the left    
    """
    assert len(s) >= k >= 0
    return s[k:] + s[0:k]

#### PICOS wrappers for LPs and SDPs

def linear_program(A, b, A_ineq=None, b_ineq=None, vtype='continuous', \
        verbose=True, solver='mosek', solv_tol = 1e-8, **params):
        #mosek_params={'infeas_report_auto': 0}):
    """ solves the general linear program of the form below
        A      *x  ==  b, 
        A_ineq *x  >=  b_ineq 
        solver options: mosek; uses the Picos wrapper.
    """
    import picos as pic; import cvxopt as cvx
    
    ## initialize Problem
    LP = pic.Problem()

    #### equality constraints...

    M,N = np.shape(A)
    ## convert numpy arrays to cvx matrices:
    Am   =   cvx.sparse(cvx.matrix( A, (M,N)) )
    bm   =   cvx.sparse(cvx.matrix( b, (M,1)) )

    ## add matrices as parameters:
    A  = pic.new_param('A',  Am )
    b  = pic.new_param('b',  bm )

    #### inequality constraint
    if A_ineq is not None:
        #print('[LP: include inequalities]')
        MM,NN = np.shape(A_ineq) #sizes
        
        ## convert numpy arrays to cvx matrices:
        A_ineqm   =   cvx.sparse(cvx.matrix( A_ineq, (MM,NN)) ) #for size 1xn arrays, replace NN by 1
        b_ineqm   =   cvx.sparse(cvx.matrix( b_ineq, (MM,1 )) )
        ## add matrices as parameters:
        A_ineq  = pic.new_param('A_ineq',  A_ineqm )
        b_ineq  = pic.new_param('b_ineq',  b_ineqm)
        

    ## define Variable to be optimised:
    x = LP.add_variable('x', N, vtype=vtype)

    ## add constraints:
    LP.add_constraint( x > 0)

    #if A is not None:    
    LP.add_constraint( A*x == b )
    if A_ineq is not None:
            assert b_ineq is not None
            LP.add_constraint( A_ineq*x >= b_ineq )

    ## set objective:
    LP.set_objective('max', 1|x) #sdp.set_objective('find', x)
    # 'find' is currently broken in Picos, see https://gitlab.com/picos-api/
    #picos/issues/124 #replacement

    if verbose: print(LP)
    
    ## solve LP
    sol = LP.solve(solver=solver, tol=solv_tol, verbose=verbose, **params)#, mosek_params=mosek_params)
    #status = sol['status']
    
    x0 = None
    try:     
        x0  = np.array(x.value)
    except:  
        x0 = 'LP: no solution or certificate returned'
    return LP, sol, x0

#### sympy tools
    
def sympy_list_to_array(A):
    return np.asarray([float(el) for el in A])


#### DISTANCE MEASURES ####

def purity(A):
    """ returns Tr(A^2) """
    return np.einsum('ij,ji->', A, A).real

def norm(A):
    """ wrapper for np.linalg.norm() """
    return np.linalg.norm(A)
    
def lp_norm(A, p):
    """ Lp-norm for Hermitian Operators
        Lp norm, BZ 13.2 eq 13.24 :math: ||A||_p =  ( 1/2 Tr|A|^p )^{1/p}
        :math: ||A|_1 = 1/2 tr(|A|) with |A| = \sqrt( A * dag(A))
    args:       p : int
    returns:    ndarray
    """
    if p == 1:
        result = 1 / 2 * np.trace(mabs(A))
    else:
        result = mt.pow(1 / 2 * np.real(np.trace(mpow(mabs(A), p))),
                     1 / p)  # real for hermitian operators, discarding (small) imaginary part
    return result

def Lp_dist(A, B, p):
    """ Lp distance
        Lp distance, BZ 13.2 eq 13.25 :math: D_p(A,B) = ||A-B||_p
    args:       A, B : ndarray
                p : int
    returns:    float
    """
    return lp_norm(A - B, p)

def HS_dist(A,B):
    """ Hilbert-Schmidt distance     
        D_HS(A,B) = ||A-B||_2^2 = Tr [ (A-B)^\dag (A-B)] 
    """
    M = A-B
    return np.trace( np.dot(dag(M), M) )
    
def trace_dist(A, B):
    """ np.trace distance
        scalar, np.trace distance :math: D_1 = ||A - B||_1
    args:       A, B : ndarray, density matrices
    returns:    float
    """
    #alt: lp_norm(A - B, 1)
    return np.sum( np.abs(np.linalg.eigvalsh(A-B)) ) / 2

def fidelity(sig, rho):
    """Fidelity
        fidelity F(\sig, \rho) = (Tr \sqrt{\sqrt{sig} rho \sqrt{sig}})^2
        BZ 13.3 Fidelity and statistical distance, eq 13.40
        accesses root_fidelity(sig, rho)
    args:       sig, rho : ndarray
    returns:    float
    """
    return root_fidelity(sig, rho) ** 2

def root_fidelity(sig, rho):
    """root fidelity
        scalar, root fidelity F(\sig, \rho) = (Tr \sqrt{\sqrt{sig} rho \sqrt{sig}})
        BZ 13.3 Fidelity and statistical distance, eq 13.41
    args:       sig, rho : ndarray
    returns:    float
    """
    print('might be unstable for certain pure states')
    sig_r = msqrt(sig)
    dist =  np.trace(msqrt(np.dot(sig_r, np.dot(rho, sig_r))))
    return dist.real  # discarding (small) imaginary part

def witness(W, rho):
    """ 
    math:       tr(W\rho)
    args:       witness, rho : ndarray
    returns:    float
    """
    overlap = np.trace(np.dot(W, rho))
    return np.real(overlap)


#### Entropy measures ####

def lin_ent(r, subs):
    """ returns linear entropy on subsystems specified in subs """
    n = number_of_qubits(r)
    all_subs = set(range(n))
    np.trace_out = complement(subs, all_subs)#list(all_subs - set(subs))
    r_red = ptrace(r, np.trace_out)
    return 2*(1-np.trace(np.dot([r_red, r_red])))    
    
def vNeumann(rho, sigma=None):
    """ von Neumann Entropy / quantum relative entropy
        scalar, von Neumann entropy of a density matrix
        :math: S = -tr(p \log p) = \sum  \lambda * log(\lambda)
        args:       p : ndarray
        returns:    float
    """
    from scipy.special import xlogy
    if sigma == None:
        rval = sp.linalg.eigvalsh(rho)
        rval = np.clip(rval, 0, 1)
        return -np.sum(xlogy(rval, rval)) # -tr(r log r)
    else:
        return QRE(rho, sigma)

def StrongSubadditivityViolation(rho, dims=[2,2,2]):
    """ returns violations of strong subadditivity
        of tripartite density matrix rho with local dimensions
        dims
        I(A:C|B) = S_AB + S_BC - S_ABC - S_B
    """
    rho_B  = ptrace(rho, [0,2], dims)
    rho_AB = ptrace(rho,  [2], dims)
    rho_BC = ptrace(rho,  [0], dims)

    S_AB = vNeumann(rho_AB)
    S_BC = vNeumann(rho_BC)
    S_ABC = vNeumann(rho)
    S_B  = vNeumann(rho_B)
    return S_AB + S_BC - S_ABC - S_B     

def WeakMonotonicityViolation(rho, dims=[2,2,2]):
    """ returns violations of strong subadditivity
        of tripartite density matrix rho with local dimensions
        dims
        S_AB + S_BC - S_A - S_C
    """
    rho_AB = ptrace(rho,  [2], dims)
    rho_BC = ptrace(rho,  [0], dims)
    rho_A  = ptrace(rho, [1,2], dims)
    rho_C  = ptrace(rho, [0,1], dims)

    S_AB = vNeumann(rho_AB)
    S_BC = vNeumann(rho_BC)
    S_A  = vNeumann(rho_A)
    S_C  = vNeumann(rho_C)
    return S_AB + S_BC - S_A - S_C     

def QRE(r, s):
    """ quantum relative entropy between r & s. only for full rank matrices.
        :math: D(rho||sigma) = tr[rho log rho] - tr[rho log sigma]
        args:
            r, s : ndarrays, density matrices
        returns:
            float,  quantum relative entropy between two states
    """
    print('stable only for full rank matrices')
    from scipy.special import xlogy
    rval     = sp.linalg.eigvalsh(r)
    sval, sU = sp.linalg.eigh(s)
    
    rval = np.clip(rval, 0, 1) # clip eigenvalues into [0,1] interval
    sval = np.clip(sval, 0, 1)
    
    Ls_log = np.diag(np.log(sval)) # diagonal eigenvalue matrix
    
    tr_r  = np.sum(xlogy(rval, rval)) # tr(r log r)
    # tr (r log s) = 
    tr_rs = np.trace( mdot([dag(sU), r, sU, Ls_log]) )
    return tr_r - tr_rs


def qRenyi(rho, a):
    """ quantum Renyi Entropy
        scalar, quantum Renyi entropy for a density matrix with integer k   
        :math:'S_{alpha}= 1/(1-alpha) \log tr p^{alpha}'
        args: p : ndarray
        args: alpha : float
        returns:    float
    """
    if a == 1:
        return vNeumann(rho)
    qrenyi_entropy = np.log(np.sum(sp.linalg.eigvalsh(rho) ** a)) / (1 - a)
    return qrenyi_entropy


#! /usr/bin/python

### Author:     Felix Huber, University Siegen
### Project:    Quantum Information, Quantum Geometry
### Module:     qcodes


from __future__ import division, print_function

from sympy import Poly
from sympy import symbols

x, y, v, w = symbols('x,y,v,w')

# from sympy.abc import x,y,v,w

from sympy.functions.combinatorial.factorials import binomial

sbinom = binomial


##############################################################################
#### ENUMERATORS ####
##############################################################################


def make_enum(x0, n, m=None):
    """ given weight distribution, create (possibly reduced) 
        enumerator polynomial of reduction having size m
    """

    if m == None:
        m = n
    W = 0
    for j in range(m + 1):
        fact = 1
        if m != n:
            fact = sbinom(m, j) / sbinom(n, j)

        W = W + x0[j] * x ** (int(n) - j) * y ** j * fact
    return W


def MacWilliams_trafo(W, q=2):
    """ given Polynomial as sympy expression, 
        permorms MacWilliams transform 
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, (x + (q ** 2 - 1) * y) / q)
    W = W.subs(w, (x - y) / q)
    return W.expand()


def primary_to_unitary(W, q=2):
    """ given Polynomial as sympy expression, 
        permorms MacWilliams transform 
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, x + y / q)
    W = W.subs(w, y / q)
    return W.expand()


def unitary_to_primary(W, q=2):
    """ given Polynomial as sympy expression, 
        permorms MacWilliams transform 
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, x - y)
    W = W.subs(w, q * y)
    return W.expand()


def Shadow_S0(W, q=2):
    W = W.subs(x, q - 1)
    W = W.subs(y, -1)
    return W


def Shadow_Nebe(W, q=2):
    """ given Polynomial as sympy expression, 
        returns Nebe shadow polynomial
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, ((q - 1) * x + (q + 1) * y) / q)
    W = W.subs(w, (y - x) / q)
    return W.expand()


def W_coeffs(W):
    """ given polynomial, extracts coefficients starting from lowest order"""
    W = W.subs(x, 1)
    c = Poly(W, y).all_coeffs()
    return c[::-1]  # returns coeffs starting with lowest order


def shadow_coeffs(W, q=2, self_dual=True):
    """ given weight distribution,
        returns coeffs of shadows) 
    """
    S1c = W_coeffs(Shadow_me(W, q))  # shadow
    S2c = W_coeffs(Shadow_Nebe(W, q))  # Nebe shadow
    return S1c, S2c


def code_purity_check(W, d, q=2, tol=1e-10):
    """ given weight distribution
        returns True if code with distance d is pure
        A_j=B_j
    """
    B = MacWilliams_trafo(W, q)
    Bj = W_coeffs(B)

    # check that A_j = B_j = 0 for j<d
    return (sum(Bj[1:d]) < tol)


def single_red_enum(A, m):
    # for AME states..
    A_mred = []
    for it in range(m + 1):
        A_mred.append(A[it] * sbinom(m, it))
    return A_mred


#### Enumerators expressed in purities

def Shadow_Nebe_in_purities(W, q=2):
    """ given Polynomial as sympy expression, 
        returns Nebe shadow polynomial
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, (x + y))
    W = W.subs(w, (y - x))
    return W.expand()


def MacWilliams_trafo_in_purities(W, q=2):
    """ given Polynomial as sympy expression, 
        permorms MacWilliams transform 
    """
    W = W.subs(x, v)
    W = W.subs(y, w)

    W = W.subs(v, (x + (q ** 2 - 1) * y) / q)
    W = W.subs(w, (x - y) / q)
    return W.expand()

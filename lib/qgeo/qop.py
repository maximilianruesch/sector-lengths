from __future__ import division, print_function

#try: 
#    from .qgeo import *
#except: 

from qgeo import *

import numpy as np
import math as m

# quantum operators: gates and states

#### GATES ####

def swap(d=2):
    """swap= d*|bell><bell|^{transpose_B}"""
    Op = dm(Bell(d=d))
    return d * ptranspose(Op, [1], d=d) 

def SWAPsqrt():
    """ square root of swap gate for qubits """
    return np.array([[2,0,0,0],
                     [0,1+1j,1-1j,0],
                     [0,1-1j,1+1j,0],
                     [0,0,0,2]]) / 2

def swap_padded(i, j, n, d=2):
    """ Swap gate, permutes particle i with paraticle j (of ket!)
        SWAP(i,j) = d*|bell><bell|_ij^{transpose_i} x Id_rest
    """
    S = [i,j]
    Sc = complement([i,j], n)
    Op = swap(d=d)
    return tensor_mix( Op, S, Id(n-2,d), Sc, d=d)


# RANDOM OBJECTS

def rand_unitary(N):
    """ N x N haar measure random unitary matrix
    
        code from:  Francesco Mezzadri - How to generate random matrices from
                    the classical compact groups". AMS Notices Vol 54, Nr 5, 2004
    """
    z   = (sp.randn(N,N) + 1j*sp.randn(N,N)) / m.sqrt(2.0)
    q,r = sp.linalg.qr(z)
    d   = sp.diagonal(r)
    ph  = d/sp.absolute(d)
    q   = sp.multiply(q,ph,q)
    return q

def rand_herm(N):
    """ Gaussian random hermitian, of dim N and full rank
        args:       N : dimension of Hilbert space
        returns:    ndarray
    """
    H = sp.randn(N,N) + 1j*sp.randn(N,N)
    return (H + dag(H)) / 2

def rand_complex(d):
    return np.random.rand(d,d) + np.random.rand(d,d) * 1j

def rand_psd(d):
    K = rand_complex(d)
    return np.dot(K, dag(K))


def rand_int_matrix(d=2, n=1, m=3, cpx=False):
    # m : int, range of entries in [-m,m]
    if complex==True:
        return np.random.randint(-m,m+1, [d**n,d**n]) + 1j* np.random.randint(-m,m+1, [d**n,d**n]) 
    return np.random.randint(-m,m+1, [d**n,d**n])

def rand_int_herm(d=2, n=1, m=3, cpx=False):
    # m : int, range of entries in [-2m,2m]
    A = rand_int_matrix(d,n,m,cpx)
    A = (A + dag(A))/2
    if trace(A) == 0:
        A = A + np.eye(d)
    return A

def rand_int_psd(d=2, n=1, m=3, cpx=False):
    # m : int, range of entries in root [-m,m]
    A = rand_int_matrix(d,n,m,cpx)
    return dot(A, dag(A))

#####

#####

CZ = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]])

def C_phase_gen(l):
    """ generalised (multi-qbit) phase gate, as used to generate hypergraph states """
    res = Id(l)
    for k in range(0,l+1):
        res = res - 2**(1-l) * (-1)**k * P_perm_u('Z'*k + 'I'*(l-k))
    return res


    
    
def C_phase_d(n, edge_list, d=2):
    """ product of generalised C_phase on subsystems in edge_list, e.g. [[0,2,3], [4]]
        for n party system of d levels each
        
        args: n : int, number of subsystems
              subs : list of ints, subsystems
              d : int, local dimension
        returns: ndarray, gate
    """
    if d==2:
        w = -1
    else:
        w  = np.exp(1j*2*np.pi/d)
    
    # start with identity (not yet in diagonal form)
    D = d**n

    C = np.ones(D, dtype=complex)
    
    # apply edges/phases
    for edge in edge_list:
        for it in range(D):
            
            b = np.base_repr(it, base=d) # e.g.'0111' #FIXME: use fast Tristan binary_rep
            b = '0'*(n-len(b)) + b       # padding
            a = string_to_array(b)       # e.g. np.array([0,1,1,1])
            # whenever there are only '1's in subsystems, 
            # introduce a phase in the Gate
            edge_prod = np.product(a[edge])
            if edge_prod != 0: #all elements nonzero
                C[it] = C[it] * w**edge_prod
    return np.diag(C)
    
def Fourier_gate(n,d):
    """Fourier gate for n systems having d-levels each.
        args: n : int, number of systems
              d : int, local dimension
        returns: array, Fourier gate
    """
    D = d**n
    w  = np.exp(2*np.pi*1j/d)

    F = np.zeros([D,D], dtype = complex)
    for it1 in range(D):
        for it2 in range(D):
            exponent = it1*it2  % d #modulo
            F[it1, it2] = w**exponent
    return F/m.sqrt(D)

# is actually slower..
#def Fourier_gate(n,d):  
#    D = d**n
#    w  = np.exp(2*np.pi*1j/d) * np.ones([D,D])
#    #calculate w**(k*l)
#    w_vec = np.arange(D) 
#    w_array = np.outer(w_vec, w_vec)
#    F= w ** w_array
#    return F/sqrt(D)

def U3_para(angles):
    """
    Parametrisation of 3x3 unitary
    J. B. Bronzan, Parametrization of SU(3)
    PhysRevD.38.1994, eq. 2.10
    args: 
        list of angles:
        t1,t2,t3 in [0,pi/2] (theta's)
        p1,...p5 in [0,2pi]  (phi's)
    returns: 
        3x3 array, unitary
    """
    [t1,t2,t3,p1,p2,p3,p4,p5] = angles
    u11 =  m.cos(t1)*m.cos(t2)*m.exp(1j*p1)
    u12 =  m.sin(t1)       *m.exp(1j*p3)
    u13 =  m.cos(t1)*m.sin(t2)*m.exp(1j*p4)
    u21 =  m.sin(t2)*m.sin(t3)*m.exp(1j*(-p4-p5)) - m.sin(t1)*m.cos(t2)*m.cos(t3)*m.exp(1j*(p1+p2-p3))
    u22 =  m.cos(t1)*m.cos(t3)*m.exp(1j*p2)
    u23 = -m.cos(t2)*m.sin(t3)*m.exp(1j*(-p1-p5)) - m.sin(t1)*m.sin(t2)*m.cos(t3)*m.exp(1j*(p2-p3+p4))
    u31 = -m.sin(t1)*m.cos(t2)*m.sin(t3)*m.exp(1j*(p1-p3+p5)) - m.sin(t2)*m.cos(t3)*m.exp(1j*(-p2-p4))
    u32 =  m.cos(t1)*m.sin(t3)*m.exp(1j*(p5))
    u33 =  m.cos(t2)*m.cos(t3)*m.exp(1j*(-p1-p2)) - m.sin(t1)*m.sin(t2)*m.sin(t3)*m.exp(1j*(-p3+p4+p5))
    U = np.array([[u11,u12,u13],
                  [u21,u22,u23],
                  [u31,u32,u33]])
    return U


#### RANDOM STATES ####
#   rand_dm, rand_dm2, rand_pure_state, rand_herm

def rand_pure_state(n, d=2):
    """ random pure state (ket) for n qubits 
        obtained by rotating '00..0' ket with random unitary
        args:       n : number of qubits
                    d : local dimension
        returns:    ndarray
    """
    return rand_unitary(d**n)[:,0]
    
def rand_pure_prod_state(n, d=2):
    psi = rand_pure_state(1, d=d)
    for it in range(n-1):
        psi = mkron([psi, rand_pure_state(1, d)])
    return psi


def rand_dm(n, d=2):
    """ random density matrix obtained by gaussian distribution of complex numbers 
        args:       n : number of qubits
                    d : local dimension
        returns:    ndarray
    """
    N = d**n
    M = np.random.normal(size=[N,N]) + 1j * np.random.normal(size=[N,N])
    rho = dot(M, dag(M))
    rho = rho/trace(rho)
    return rho

def rand_rankk_dm(n,d,k):
    """ random rank-k density matrix, obtained by convex combination of k random pure states """
    r = np.sum([dm(rand_pure_state(n,d)) for el in range(k)], axis=0) 
    return r/trace(r)

def rand_bisep(d,k):
    # poor man's random biseparable state sampling.
    # tripartite d-level state, each of rank k
    II = np.eye(d)
    r1 = dm(rand_rankk_dm(2,d,k))
    r2 = dm(rand_rankk_dm(2,d,k))
    r3 = dm(rand_rankk_dm(2,d,k))
    rAB = mkron([r1, II])
    rAC = tensor_mix(r2, [0,2],II,[1])
    rBC = mkron([II,r3])
    r = rAB + rAC + rBC
    return r / trace(r)

def rand_dm2(n, m=2, d=2):
    """ random density matrix obtained by tracing out a pure system 
        for of n+m qubits. default: m=2
        fixme: implement a faster algorithm. e.g. one of 
        http://quantumspy.wordpress.com/2014/10/24/generating-random-density-matrices/
    """
    m = n
    psi = rand_pure_state(n + m, d=d) #pure state of n+m qubits
    # has dimension 2**N
    rho = dm(psi)
    return ptrace(rho, np.arange(n, n+m), d=d)

def rand_complex(d):
    """ "random" complex matrx """
    return np.random.rand(d,d) + np.random.rand(d,d) * 1j

#### STATES #####

### n particle states (kets)
#   GHZ, GHZ_m
def GHZ(n=3, d=2):
    """ n qubit GHZ state (ket). 
        Guehne - Entanglement Detection, eq 54 
    """
    psi = 0
    for level in range(d):
        psi = psi + ket(str(level) * n, d)
    return psi / sqrt(d)

def GHZ_m(n=3):
    """ n qubit GHZ minus state (ket), for qubits only (d=2)
        Guehne - Entanglement Detection, eq 54
    """
    return (ket('0'*n) - ket('1'*n)) / sqrt(2)
    
    
def dicke(n,k):
    """ returns ket of dicke state of n qubits with k excitations ('1') """
    if not n>=k: raise Exception( "n should be bigger than k!" )
    s = k*'1' + (n-k)*'0'
    s_list = string_permutations_unique(s)
    psi = np.sum([ket(el) for el in s_list], axis=0)
    return ket_normalise(psi)

def W_state(n):
    return dicke(n,1)

### 1 qubit states


def ket_theta_phi(theta, phi):
    """ returns the ket
        |theta,phi> = cos(theta/2) |0> + e^(i phi) sin(theta/2) |1>
    """
    return np.cos(theta/2.0)*ket('0') + np.sin(theta/2.0)*np.exp(1.0j*phi)*ket('1')



### 2 qubit states
    
##  Bell states (kets)
def psi_p():
    return (ket('00') + ket('11')) / sqrt(2)
def psi_m():
    return (ket('00') - ket('11')) / sqrt(2)
def phi_p():
    return (ket('01') + ket('10')) / sqrt(2)
def phi_m():
    return (ket('01') - ket('10')) / sqrt(2)

def Bell(d=2):
    """qudit Bell state"""
    psi = np.zeros(d**2)
    for level in range(d):
        #psi = psi + ket(str(level) * 2, d)
        psi = psi + mkron([single_ket(level, d), single_ket(level, d)])
    return psi / sqrt(d)

def gamma():
    """ Bell Type 2-qubit state (ket)
        |gamma> = (|00> + |01> - |10> + |11>) / 2     
    """
    return ( ket('00') + ket('01') - ket('10') + ket('11') ) / 2
    
def gamma_bar():
    """ Bell Type 2-qubit state (ket)
        |gamma_bar> = (|00> - |01> + |10> + |11>) / 2 
    """
    return ( ket('00') - ket('01') + ket('10') + ket('11') ) / 2




### 4 qubit states
#   from Physics Reports: Guehne, Toth - Entanglement Detection


#Todo: normalisation wrong. check with literature
def singlet4(): 
    """ 4 qubit singlet ket. p 21, eq 61 (ket)"""
    psi = ( ket('0011') + ket('1100') \
    - 1/2* np.kron( ket('01') + ket('10'), ket('01') + ket('10') ) ) / sqrt(3)
    return psi


def chi4():
    """ 4 qubit chi ket. p 22, eq 664 (ket)
        Osterloh/Siewert - Entanglement monotones and maximally entangled states in multipartite qubit systems
        quant-ph/0506073, eq. 16
    """
    chi = ( sqrt(2) * ket('1111') + ket('0001') + ket('0010') + ket('0100') + ket('1000') ) / sqrt(6)
    return chi

    
def HS4():
    """ 4 qubit Higuchi Sudbery state
        Higuchi A and Sudbery A 2000 How entangled can two couples get?
        Phys. Lett. A 272 213-7, section 3.
        Iain D K Brown et al., Searching for highly entangled multi-qubit states, 
        J. Phys. A: Math. Gen. 38 (2005) 1119-1131, eq. 13
    """
    w = -1/2 + sqrt(3)/2 * 1j  #np.exp( 2*np.pi*1j / 3 ) 
    hs = ( ket('0011') + ket('1100') + w*(ket('1010') + ket('0101')) + w**2 *(ket('1001') + ket('0110')) ) / sqrt(6)
    return hs

def Brown4():
    """ 4 qubit highly entangled state (maximises bipartite NPT) 
        Brown et al., 2005 J. Phys. A: Math. Gen. 38 1119, eq. 16
        Searching for highly entangled multi-qubit states
    """
    return 1/2*(ket('0000') + ket('+011') + ket('1101') + ket('-110'))


def M4():
    """ Gour, Wallach - All Maximally Entangled Four Qubits States
        arxiv: 1006:0036, eq.2
    """
    u0 = mkron_self(phi_p(), 2)
    u1 = mkron_self(phi_m(), 2)
    u2 = mkron_self(psi_p(), 2)
    u3 = mkron_self(psi_m(), 2)
    M = 1j/sqrt(2) * u0 + 1/sqrt(6) * (u1 + u2 + u3)
    return M


### Four-partite states

def phi_34():
    """Maximizing the geometric measure of entanglement
        Steinberg, Guhne
        arxiv.org/abs/2210.13475
        AME and has high geometric measure of entanglement
    """
    return (ket('022',d=4) + ket('033',d=4) + ket('120',d=4) + \
            ket('131',d=4) + ket('212',d=4) + ket('203',d=4) + \
            ket('310',d=4) + ket('301',d=4) ) / (sqrt(2)*2)
    
    

### hypergraph states: 
#   Guehne et al. - Entanglement and nonclassical properties of hypergraph state, arXiv:1404.6492v2
def hypergraph_V3():
    """ four qubit hypergraph V3 state (ket)
        Guehne et al. - Entanglement and nonclassical properties of hypergraph state, arXiv:1404.6492v2
        eq. 21    
    """
    v3 = 1/sqrt(8)*( ket('0011') + ket('0101') + ket('0110') + ket('1001') + ket('1010') + ket('1100') \
    + ket('0000') - ket('1111') )
    return v3
    
def hypergraph_V9():
    """ four qubit hypergraph V9 state (ket)
           Guehne et al. - Entanglement and nonclassical properties of hypergraph state, arXiv:1404.6492v2
        eq. 22    
    """
    v9 = GHZ_m(4) / sqrt(2) + ( np.kron( ket('01'), gamma() ) + np.kron( ket('10'), gamma_bar() ) ) / 2
    return v9
    
def hypergraph_V14():
    """ four qubit hypergraph V14 state (ket)
        Guehne et al. - Entanglement and nonclassical properties of hypergraph state, arXiv:1404.6492v2
        eq. 24   
    """
    v14 = 1/sqrt(8)*( ket('0011') + ket('0101') + ket('0110') + ket('1001') + ket('1010') + ket('1100') \
    + ket('0001') - ket('1110') )
    return v14

def hypergraph_S1():
    """ six qubits hypergraph state S1 """
    # typo in Guhne paper eq 26, sqrt missing
    S1 = 1/sqrt(8)*( ket('000000') - ket('111111') \
        + np.kron( (ket('001') + ket('010') + ket('100')), ket('000') ) \
        + np.kron( (ket('110') + ket('101') + ket('011')), ket('111') ) )
    return S1

### Osterloh / Siewert states (comb monotones)
def OS_psi4():
    """ 5 qubit Osterloh Siewert state eq 24 """
    return (ket('11111') + ket('11100') + ket('00010') + ket('00001')) / 2
    
def OS_psi5():
    """ 5 qubit Osterloh Siewert stateeq 25 """
    return (sqrt(2) * ket('11111') + ket('11000') + ket('00100') \
    + ket('00010') + ket('00001')) / sqrt(6)
    
def OS_psi6():
    """ 6 qubit Osterloh Siewert state eq 26 """
    return (sqrt(3) * ket('11111') + ket('10000') + ket('01000') \
         + ket('00100') + ket('00010') + ket('00001')) / (2*sqrt(2))

#AME approximations
def Fano7():
    """ 7 qubit Fano state, having 32 out of 35 marginals 
        maximally mixed, satisfying bound on how many
        3-RDM can be at most mixed 
        corresponding graph is the fano plane
    """
    h= [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0],\
        [0,6], [1,6], [2,6], [3,6], [4,6], [5,6],\
        [1,3], [3,5], [5,1]]
    return hypergraph_state(7, h)

    
### n particle states (density matrices)
#   identity, mmix
def identity(n, d=2):
    """ identity operator for n-qubits
        args: n: int, number of qubits
        return:  ndarray
    """
    return np.eye(d ** n)
    
def Id(n, d=2):
    """ alias for identity()
        identity operator for n-qubits
        args: n: int, number of qubits
        return:  ndarray
    """
    return identity(n, d=d)
    
def mmix(n, d=2):
    """ maximally mixed n-qubit state
        args:       n : int
        returns:    ndarray 
    """
    return np.eye(d ** n) / d ** n




### ring cluster states
def RCL(n):
    """generate ring cluster density matrix of n>=3 qubits
        arg:       int n : number of qubits
        returns:   ndarray
        ## FIXME: rewrite nicer
    """
    ia  = np.array([Z, X, Z])
    #first generator g0, all others are obtained by shifting that one
    if n == 3: #n = 3 too short to produce list of identities.
        return RCL3()
    else:
        ids = np.array( [I, ] * (n - 3) )
        g0 = np.vstack((ia, ids))
    Id = np.eye(2**n)
    R = Id #initialise state

    # loop over interactions, shift by 4 as qubit are 2x2 matrices
    for k in range(n):
        R = dot(R, mkron(np.roll(g0, 4*k)) + Id)
    return R / 2**n #normalise

def RCL3():
    ### needed for RCL(n)
    G = ['XZZ', 'ZXZ', 'ZZX']
    return graph_state(G)
def RCL4():
    G = ['XZIZ', 'ZXZI', 'IZXZ', 'ZIZX']
    return graph_state(G)
def RCL5():
    G=['XZIIZ', 'ZXZII', 'IZXZI', 'IIZXZ', 'ZIIZX']
    return graph_state(G)
    
def RCL4_lin():
    """ ring cluster state for 4 qubits
        _____________

        1   2   3   4
    """
    G = ['XZII','ZXZI','IZXZ','IIZX']
    return graph_state(G)
    
def RCL4_lin_perm_23():
    """ linear cluster state for 4 qubits, having a funny layout
        the particles connected: 1-3-2-4
         ______
        /   ___\  
            \______/
        
        1   2   3   4
    """
    G = ['XIZI','IXZZ','ZZXI','IZIX']
    return graph_state(G)
    
def RCL4_lin_perm_12_34():
    """ linear cluster state for 4 qubits, having a funny layout
        the particles connected: 1-3-2-4
         ___________
        /           \    
        \___/   \___/
        
        1   2   3   4
    args:       none
    returns:    ndarray
    """
    G = ['XZIZ','ZXII','IIXZ','ZIZX']
    return graph_state(G)
    
def RCL6_():
    """ 6 qubit cluster state
         1__2
       0 /__\ 3
         \__/
         5  4  
    """
    G = ['XZIZIZ','ZXZIII','IZXZII','ZIZXZI','IIIZXZ','ZIIIZX']
    return graph_state(G)

    
def RCL6__():
    """ 6 qubit graph state /  AMES state
        has maximally mixed 3-RDMs (reduced density matrices)
         1____2
       0 /|__|\ 3
         \|__|/
         5    4 

    """
    G = ['XZIZIZ','ZXZIIZ','IZXZZI','ZIZXZI','IIZZXZ','ZZIIZX']
    return graph_state(G)
    
def RCL6_nn():
    """ 6 qubit ring cluster cluster state, nearest and next nearest connected        
    """
    G = ['XZZIZZ','ZXZZIZ','ZZXZZI','IZZXZZ','ZIZZXZ','ZZIZZX']
    return graph_state(G)


def AME52():
    return RCL(5)

def AME62():
    """ absolute maximally entangled state on 6 qubits.
        PHYSICAL REVIEW A 92, 032316 (2015)
        Absolutely maximally entangled states, combinatorial designs, and multiunitary matrices
    """
    G = ['XZZXII','IXZZXI','XIXZZI','ZXIXZI','XXXXXX','ZZZZZZ']
    return graph_state(G)
    
def AME62_v2():
    """ absolute maximally entangled state on 6 qubits as defined by 
        graph
    """
    h = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],
          [0,3],[1,5],[2,4]]
    return hypergraph_state(6,h)

def cube_graph_state():
    """ returns ket of eight qubit cube graph state, having max mixed 3-RDMs """
    edges = [[0,1], [0,2], [0,4], \
             [1,3], [1,5],\
             [2,3], [2,6],\
             [3,7],\
             [4,5], [4,6],\
             [5,7],\
             [6,7]]
    return hypergraph_state(8, edges)


def AME43_U():
    In = ['00', '01', '02', '10', '11', '12','20', '21', '22']
    Out= ['00', '11', '22', '12', '20', '01','21', '02', '10']
    U_ame = 0
    for j in range(9):
        U_ame = U_ame + ketbra(ket(In[j], d=3), ket(Out[j], d=3))
    return U_ame

### qutrit state

def AME43():
    """ returns AME(4,3) ket. latin square formulation from Goyeneche et al."""
    return 1/3*(ket('0000', d=3) + ket('0112', d=3) + ket('0221', d=3) +\
        ket('1011', d=3) + ket('1120', d=3) + ket('1202', d=3) +\
        ket('2022', d=3) + ket('2101', d=3) + ket('2210', d=3))
        
def AME43_graph_state():
    """ AME 43 from graph  with double edge"""
    #old:
    #op0 = C_phase_d(4, 3, 0, 1);     #op1 = C_phase_d(4, 3, 1, 2)
    #op2 = C_phase_d(4, 3, 2, 3);     #op3 = C_phase_d(4, 3, 3, 0)
    #gates = mdot([op0, op0, op1, op2, op3]) # op0 is a double edge
    
    #initial state to apply the gates  |0> bar
    zerobar = mdot( [Fourier_gate(4,3), ket('0000', d=3)] )
    #edges = [[0,1], [0,1], [1,2], [2,3], [3,0]]
    # for AME43 Bell inequality investigation: double gate at last pair (2,3)
    edges = [ [0,1], [1,2], [2,3], [3,0], [2,3]] 
    C = C_phase_d(4, edges, d=3)    
    AME43_g = mdot([C, zerobar]) # apply to initial state
    return AME43_g

def AME46():
    """
    2020-09-02 
    2021-01-11

    Suhail Ahmad Rather
    Adam Burchardt
    Wojciech Bruzda
    Grzegorz Rajchel-Mieldzioc
    Arul Lakshminarayan
    Karol Zyczkowski

    arXiv:2104.05122
 
    ------------------------------------------------------------------------
 
    it returns U such that
    svd(U) = svd(U^R) = svd(U^G) = [1 1 1 ... 1] = vector of 36 ones
    where ^R := reshuffling
          ^G := partial transpose (w.r.t. 2nd subsystem)

    in other words
    it returns a complex representation of the AME(4,6) state
    """

    a = 1 / np.sqrt(5.0 + np.sqrt(5.0))
    b = np.sqrt(5.0 + np.sqrt(5.0)) / np.sqrt(20.0)
    c = 1 / np.sqrt(2.0)
    w = np.exp(1j * np.pi / 10.0)
    U = np.zeros((6**2, 6**2), dtype = np.complex)

    # 20 non-zero entries of 1st six rows:
    U[1, 1] = c*(w**3);
    U[5, 0] = c/w
    U[0, 7] = b/w;
    U[2, 6] = a/(w**7);
    U[3, 6] = b*(w**2);
    U[5, 7] = a/(w**3)
    U[0, 15] = a/(w**2);
    U[1, 14] = b*(w**9);
    U[4, 14] = a*(w**5);
    U[5, 15] = b*(w**6)
    U[2, 20] = a/(w**6);
    U[3, 20] = b/w;
    U[4, 21] = c/w
    U[1, 29] = a;
    U[2, 28] = b*(w**7);
    U[3, 28] = a*(w**2);
    U[4, 29] = b*(w**6)
    U[0, 34] = c*(w**6);
    U[2, 35] = b/(w**5);
    U[3, 35] = a/(w**6)

    # 20 non-zero entries of 2nd six rows:
    U[6, 0] = c
    U[10, 1] = c/(w**4)
    U[6, 7] = a*(w**8)
    U[8, 6] = b/(w**8)
    U[9, 6] = a/(w**9)
    U[11, 7] = b/(w**4)
    U[6, 15] = b/(w**3)
    U[7, 14] = a*(w**6)
    U[10, 14] = b/(w**8)
    U[11, 15] = a/(w**5)
    U[8, 20] = b*w
    U[9, 20] = a/(w**4)
    U[7, 21] = c*(w**10)
    U[8, 28] = a/(w**4)
    U[9, 28] = b*w
    U[7, 29] = b*(w**7)
    U[10, 29] = a*(w**3)
    U[8, 35] = a/(w**8)
    U[9, 35] = b*w
    U[11, 34] = c/(w**7)

    # 18 non-zero entries of 3rd six rows:
    U[13, 5] = b/(w**3)
    U[14, 4] = b/(w**4)
    U[15, 4] = a/w
    U[16, 5] = a*(w**9)
    U[12, 11] = a*(w**4)
    U[13, 10] = a*(w**6)
    U[16, 10] = b*(w**8)
    U[17, 11] = b
    U[15, 13] = c/(w**5)
    U[16, 12] = c*w
    U[14, 18] = c;
    U[17, 19] = c/(w**6)
    U[12, 26] = c*(w**8)
    U[14, 27] = a*(w**6)
    U[15, 27] = b/w
    U[12, 33] = b/(w**6)
    U[13, 32] = c/(w**6)
    U[17, 33] = a

    # 18 non-zero entries of 4th six rows:
    U[19, 5] = a/(w**3)
    U[20, 4] = a/(w**8)
    U[21, 4] = b*(w**5)
    U[22, 5] = b/w
    U[18, 11] = b*(w**2)
    U[19, 10] = b/(w**4)
    U[22, 10] = a*(w**8)
    U[23, 11] = a*(w**8)
    U[19, 12] = c/w
    U[20, 13] = c/(w**2)
    U[18, 19] = c*(w**6)
    U[21, 18] = c/w
    U[23, 26] = c*(w**2)
    U[20, 27] = b/(w**8)
    U[21, 27] = a/(w**5)
    U[18, 33] = a*(w**2)
    U[22, 32] = c*(w**6)
    U[23, 33] = b/(w**2)

    # 18 non-zero entries of 5th six rows:
    U[24, 2] = a
    U[26, 3] = a*w
    U[27, 3] = b*(w**2)
    U[29, 2] = b
    U[25, 9] = b*w
    U[26, 8] = b*w**3
    U[27, 8] = a/(w**6)
    U[28, 9] = a*w
    U[24, 16] = b/(w**4)
    U[26, 17] = c/(w**5)
    U[29, 16] = a*(w**6)
    U[24, 23] = c/(w**7)
    U[25, 22] = a/(w**2)
    U[28, 22] = b*(w**8)
    U[25, 25] = c/(w**6)
    U[29, 24] = c
    U[27, 31] = c/(w**2)
    U[28, 30] = c/(w**6)

    # 18 non-zero entries of last six rows:
    U[30, 2] = b/(w**9)
    U[32, 3] = b
    U[33, 3] = a/(w**9)
    U[35, 2] = a*w
    U[31, 9] = a/(w**6)
    U[32, 8] = a/(w**8)
    U[33, 8] = b/(w**7)
    U[34, 9] = b*(w**4)
    U[30, 16] = a/(w**3)
    U[33, 17] = c/(w**5)
    U[35, 16] = b/(w**3)
    U[31, 22] = b*w
    U[34, 22] = a*w
    U[35, 23] = c*(w**4)
    U[30, 24] = c*w
    U[34, 25] = c*(w**7)
    U[31, 30] = c/(w**3)
    U[32, 31] = c*(w**6)
    
    amestate = np.zeros(6**4, dtype=np.complex)
    for i in range(6):
        for j in range(6):
            ketij = ket(str(i)+str(j), d=6)
            psiij = dot(U, ketij)
            amestate += mkron([ketij, psiij])
    amestate = amestate / 6
    return amestate

def Werner(a):
    """ qubit Werner state, page 27, eq 2.35

      P. Krammer - Quantum Entanglement: Detection,Classification, and Quantification 
      https://homepage.univie.ac.at/reinhold.bertlmann/pdfs/dipl_diss/Krammer_Diplomarbeit.pdf
   
    -1/3 <= a <= 1    state normalization
     1/3 <  a <= 1    <=>   Werner state is entangled (by PPT criterium)
    
    """
    assert -1/3 <= a <= 1
    psi_m = (ket('01') - ket('10')) /sqrt(2)
    
    return a*dm(psi_m)  + (1-a)/4*np.eye(4)


def Isotropic(a,d=2):
    """ qubit Isotropic state, page 10, eq. 2.19

      P. Krammer - Quantum Entanglement: Detection,Classification, and Quantification 
      https://homepage.univie.ac.at/reinhold.bertlmann/pdfs/dipl_diss/Krammer_Diplomarbeit.pdf
   
    -1/(d**2-1) <= a <= 1    state normalization
     1/(d+1)    <  a <= 1      <=>   Isotropic state is entangled (by PPT criterium)
    
    """
    assert -1/(d**2-1) <= a <= 1
    return a*dm(Bell(d=d))  + (1-a)/d**2 *np.eye(d**2)



def Hyllus(): 
    """ Hyllus state 
        BI-SEP but not SEP.
    """
    ## corresponding paper

    eta = sqrt(3/2)
    Hy = 1/(3+2*eta+3/eta) *\
    np.array([[2*eta,0,0,0,0,0,0,0],\
              [0,1,1,0,1,0,0,0],\
              [0,1,1,0,1,0,0,0],\
              [0,0,0,1/eta,0,0,0,0],\
              [0,1,1,0,1,0,0,0],\
              [0,0,0,0,0,1/eta,0,0],\
              [0,0,0,0,0,0,1/eta,0],\
              [0,0,0,0,0,0,0,0]])
    return Hy

try: 
    from .qgeo import *
    from .qop import *
except: 
    from qgeo import *
    from qop import *
    


### channel operations

def ketbra_idx(i,j,d):
    """ returns |i><j| """
    rho = np.zeros((d,d), dtype=np.complex)
    rho[i,j] = 1.0
    
    return rho

def tensor_channels(M,N, d=2):
    """ returns the channel M \otimes N """
    def MtensorN(rho):
        rho2 = np.zeros( (d*d,d*d), dtype=np.complex)
        
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        
                        rho2+= rho[i*d+j,k*d+l] * mkronf([M(ketbra_idx(i,k,d=d)), N(ketbra_idx(j,l,d=d))])
        return rho2
    return MtensorN

def composite(ch1, ch2):
    """ returns the channel M(rho) = ch1(ch2(rho)). Thus, ch2 is applied first! """
    def ch12(rho):
        return ch1(ch2(rho))
    
    return ch12



def channel_to_ellipsoid(channel):
    """ returns the ellipsoid representation of the qubit channel, i.e. a displacement vector k and a 3 by 3 matrix L,
        s.t. the action of the channel is given by mapping rho=1/2(Id + v sigma) to 1/2(Id + (Lv + k) sigma)
    """
    displ = pauli_coeff(channel(0.5*Id(1)))
    kx = displ['X']
    ky = displ['Y']
    kz = displ['Z']
    
    L = np.zeros((3,3))
    
    rx = 0.5*(Id(1) + X)
    ry = 0.5*(Id(1) + Y)
    rz = 0.5*(Id(1) + Z)
    
    displ = pauli_coeff(channel(rx))
    L[0,0] = displ['X'] - kx
    L[1,0] = displ['Y'] - ky
    L[2,0] = displ['Z'] - kz
    
    displ = pauli_coeff(channel(ry))
    L[0,1] = displ['X'] - kx
    L[1,1] = displ['Y'] - ky
    L[2,1] = displ['Z'] - kz
    
    displ = pauli_coeff(channel(rz))
    L[0,2] = displ['X'] - kx
    L[1,2] = displ['Y'] - ky
    L[2,2] = displ['Z'] - kz
    
    k = np.array([kx,ky,kz])
    
    return k, L

def ellipsoid_to_channel(k, L):
    """ returns the channel corresponding to the ellipsoid defined by the displacement vector k and a 3 by 3 matrix L,
        s.t. the action of the channel is given by mapping rho=1/2(Id + v sigma) to 1/2(Id + (Lv + k) sigma)
    """
    def chan(rho):
        ps = pauli_coeff(rho, [1])
        v = np.array([ps['X'], ps['Y'], ps['Z']])
        
        vp = dot(L,v) + k
        
        return 0.5*(Id(1) + vp[0]*X + vp[1]*Y + vp[2] * Z)
    
    return chan



def choi(channel, d=2):
    """ returns the choi state of the channel M(i.e. it is normalized!),
        defined by eta = Id \otimes M (|Phi+>)
    """
    phip = ket('0', d=d*d)
    
    for i in range(d):
        phip[i*d+i] = 1.0
    phip = dm(ket_normalise(phip))
    
    eta = tensor_channels(identity_channel, channel, d=d)(phip)
    
    return eta


def choi_to_channel(choi, d=2):
    """ returns the channel corresponding to the normalized choi state, i.e.
        M(rho) = d trace( dot(rho^T \otimes Id, choi) )
    """
    def channel(rho):
        return d*ptrace(dot(mkronf([rho.transpose(), Id(1,d=d)]), choi), [0], d=d)
    return channel


### basic qubit channels


def identity_channel(rho):
    return rho

def dephasing(rho):
    """ returns the diagonal part of rho"""
    n = number_of_qubits(rho)
    rho2 = np.zeros((2**n,2**n), dtype=np.complex)
    for i in range(2**n):
        rho2[i,i] = rho[i,i]
    
    return rho2

def ampdamp(gamma):
    def channel(rho):
        K1 = np.array([[1,0],[0,sqrt(1-gamma)]], dtype = np.complex)
        K2 = np.array([[0,sqrt(gamma)],[0,0]], dtype = np.complex)
        
        return mdot([K1, rho, dag(K1)]) + mdot([K2, rho, dag(K2)]) 
    
    return channel
    

def phase_flip(p):
    return ellipsoid_to_channel(np.array([0,0,0]), np.diag([1-p,p,p]))

def planar(s,q):
    return ellipsoid_to_channel(np.array([0,0,0]), np.diag([0,s,q]))

### drawing channels

def draw_channel(channel):
    k, l = channel_to_ellipsoid(channel)
    o1, ld, o2h = np.linalg.svd(l)

    return draw_channel_p(k[0], k[1], k[2], ld[0], ld[1], ld[2], o1, dag(o2h))

def draw_channel_p(kx,ky,kz,lx,ly,lz, o1=None, o2=None):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if o1 == None:
        o1 = np.identity(3)
    if o2 == None:
        o2 = np.identity(3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    
    u, v = np.mgrid[0:2*np.pi:13j, 0:np.pi:11j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    
    x2 = lx*np.cos(u)*np.sin(v)+kx
    y2 = ly*np.sin(u)*np.sin(v)+ky
    z2 = lz*np.cos(v)+kz
    
    for i in range(len(x2)):
        for j in range(len(x2[i])):
            v = np.array([x[i][j], y[i][j], z[i][j]])
        
            vp = mdot([o1,np.diag([lx,ly,lz]),dag(o2), v]) + np.array([kx,ky,kz])
            x2[i][j] = vp[0]
            y2[i][j] = vp[1]
            z2[i][j] = vp[2]
    
    

    ax.plot_wireframe(x, y, z, color="#000000", linewidth=0.2)
    ax.plot_wireframe(x2, y2, z2, color="r", linewidth=0.3)
    ax.plot([0, kx], [0, ky], [0, kz], color='red', alpha=0.8, lw=2, ls="-")
    
    vecx = dot(o1, np.array([lx,0,0]))
    vecy = dot(o1, np.array([0,ly,0]))
    vecz = dot(o1, np.array([0,0,lz]))
    ax.plot([kx, kx+vecx[0]], [ky, ky+vecx[1]], [kz, kz+vecx[2]], color='green', alpha=0.8, lw=3)
    ax.plot([kx, kx+vecy[0]], [ky, ky+vecy[1]], [kz, kz+vecy[2]], color='blue', alpha=0.8, lw=3)
    ax.plot([kx, kx+vecz[0]], [ky, ky+vecz[1]], [kz, kz+vecz[2]], color='black', alpha=0.8, lw=3)

    ax.scatter([0], [0], [0], c="g", s=20)
    
    return fig
    

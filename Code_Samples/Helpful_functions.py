from sympy.combinatorics.permutations import Permutation 
from scipy.spatial.transform import Rotation as R
import itertools
import numpy as np
"""
Author: Chris DuPre
This is a script to contain several useful functions for other projects. 

"""

def levi_cevita_tensor():
    """
    Function to form Levi-Cevita Tensor

    Parameters:
    ----------
    None

    Outputs
    -------
    LCT: np.array((3,3,3)), Levi-Cevita Tensor

    """
    LCT = np.zeros((3,3,3))
    perms = itertools.permutations(range(3))
    for perm in perms:
        if Permutation(perm).is_even:
            LCT[perm] = 1
        else:
            LCT[perm] = -1
    return LCT


def hat(v:np.array,LCT:np.array=None):
    """
    Returns hat(v) in s0(3) [Lie algebra]. Defined as 
    the unique mapping v: R^3 -> \R^{3x3} such that
        hat(v)w = vxw.
    
    Parameters:
    ----------
    v: np.array (3): Vector input
    LCT: np.array (3,3,3): Levi-Cevita Tensor. Default is none which
        causes the function to create a Levi-Cevita Tensor on the
        fly

    Outputs:
    -------
    hat(v): np.array (3,3)

    """
    if np.any(LCT==None):
        LCT = levi_cevita_tensor()
    vhat = np.einsum("ijk,j",LCT,v)
    return vhat

def so3_vec(M:np.array):
    """
    Inverse map of hat. Take a matrix representation of so(3)
    and returns the vector representation in R^3

    Parameters:
    -----------
    M: np.array: Member of so(3)

    Outputs:
    -------
    v: Vector in R^3

    """
    return np.array([-M[1,2],M[0,2],-M[0,1]])


def sample_SO3():
    """
    Algorithm is from https://arxiv.org/abs/math-ph/0609050 
    combined with the mapping S:O(n) -> SO(n)
    """
    G = np.random.randn(3,3)
    Q,R = np.linalg.qr(G)
    R_diag = np.diag(R)
    Lambda = np.diag(R_diag/np.abs(R_diag))
    Q = np.conjugate(Lambda)@Q
    if np.linalg.det(Q) < 0:
        Q_copy = Q.copy()
        #Exchange rows
        Q_copy[0,:] = Q[1,:]
        Q_copy[1,:] = Q[0,:]
        return Q_copy
    else:
        return Q
    

def sample_unit_quaternion():
    """
    Unit Quaternions can be identified with S^3 (unit sphere in \R^4)
    Thus we can sample unit sphere according to a Gaussian-rescale strategy
    """
    G = np.random.randn(4)
    G = G/np.linalg.norm(G)
    return G
    
def quaternion_to_SO3(q:np.array):
    """
    Maps quaternions to SO(3) by first identifing H with SU2 as in
    https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf
    and then using the homomorphism defined in: 
    https://www.math.nagoya-u.ac.jp/~richard/teaching/f2022/SML_Tom_Yesui.pdf
    This could probably be done in a cleaner way but I don't
    know enough about quaternions to say for sure

    Parameters:
    ----------
    q: unit quaternion expressed as an element of R^4

    Output:
    -------
    M: np.array: Representative in SO(3)
    """

    pauli_1 = np.array([[0,1],[1,0]],dtype=complex)
    pauli_2 = np.array([[0,-1j],[1j,0]],dtype=complex)
    pauli_3 = np.array([[1,0],[0,-1]],dtype=complex)
    pauli_arr = [pauli_1,pauli_2,pauli_3]
    U = q[0]*np.eye(2)+1j*(q[1]*pauli_3+q[2]*pauli_2+q[3]*pauli_1)
    UH = np.conjugate(np.transpose(U))
    if not np.allclose(U@UH,np.eye(2)):
        raise ValueError("Something is wrong with the quaternion. Make sure all entries are real.")
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M[i,j] = np.trace(pauli_arr[i]@U@pauli_arr[j]@UH)/2

    return M


def project_SO3(M:np.array):
    """
    Finds the matrix S such that S\in SO(3)
    and S = argmin_{U\in SO(3)}\|U-M\|_{F}
    using Kabsch algorithm as described in 
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters:
    ----------
    M: np.array: Arbitrary 3x3 matrix
    
    Outputs:
    -------
    S: np.array: Projection of M onto SO(3)
    """
    U,_,Vh = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U@Vh))
    s = np.array([1,1,d])
    return U@np.diag(s)@Vh

def SO3_from_Euler(roll,pitch,yaw):
    return R.from_euler(angles=[yaw,pitch,roll],seq="ZYX").as_matrix().T

def Euler_from_SO3(M):
    [yaw,pitch,roll] = R.from_matrix(M.T).as_euler(seq="ZYX")
    return [roll,pitch,yaw]


def delay_embdedding(X,d):
    m,n = X.shape
    D = np.zeros((d*m,n+1-d))
    for k in range(n+1-d):
        sub = X[:,k:k+d]
        D[:,k] = sub.T.flatten()
    return D

def prune_A(A,radius):
    eigvals, eigvecs = np.linalg.eig(A)
    indices = np.argwhere(np.abs(eigvals) <=radius).flatten()
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:,indices]
    dual = np.linalg.pinv(eigvecs)
    return eigvecs@np.diag(eigvals)@dual

def project_A(A,radius):
    eigvals, eigvecs = np.linalg.eig(A)
    above_indices = np.argwhere(np.abs(eigvals) >=radius).flatten()
    above_vals = eigvals[above_indices]
    proj_vals = np.array([val/abs(val)*radius for val in above_vals])
    eigvals[above_indices] = proj_vals
    dual = np.linalg.pinv(eigvecs)
    return eigvecs@np.diag(eigvals)@dual

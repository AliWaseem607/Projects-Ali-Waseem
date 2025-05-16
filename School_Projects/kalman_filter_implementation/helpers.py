import numpy as np
from scipy.linalg import expm

def divide_B_mat(B):
    n = int(B.shape[0]/2)
    B_11 = B[:n,:n]
    B_12 = B[:n,n:]
    B_21 = B[n:,:n]
    B_22 = B[n:,n:]
    return B_11, B_12, B_21, B_22

def get_F_mat(f1_m, f2_m, a, F22):
    F11 = np.array([[0,0,0,0,0],
                    [-f2_m,0,0,0,0],
                    [f1_m,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,0,0]])
    F12 = np.array([[1,1,0,0],
                    [0,0,np.cos(a), -np.sin(a)],
                    [0,0,np.sin(a), np.cos(a)],
                    [0,0,0,0],
                    [0,0,0,0]])
    F21 = np.zeros((F22.shape[0],F11.shape[1]))

    F = np.vstack([
        np.hstack([F11,F12]),
        np.hstack([F21,F22])
    ])

    return F


def get_G_mat(a):
    G11 = np.array([[1,0,0],
                    [0,np.cos(a), -1*np.sin(a)],
                    [0,np.sin(a), np.cos(a)],
                    [0, 0, 0],
                    [0, 0, 0]])
    G12 = np.zeros((5,3))
    G21 = np.zeros((4,3))
    G22 = np.array([[0,0,0],
                    [1,0,0],
                    [0,1,0],
                    [0,0,1]])
    G = F = np.vstack([
        np.hstack([G11,G12]),
        np.hstack([G21,G22])
    ])
    
    return G

def get_Phi_Q_mat(F, G, W, dt):

    A = np.vstack([
        np.hstack([F*-1, G @ W @ G.T]),
        np.hstack([np.zeros(F.shape), F.T])]) * dt

    B = expm(A)
    _, B_12, _, B_22 = divide_B_mat(B)

    Phi = B_22.T
    Q = Phi @ B_12
    return Phi, Q



def random_bias(val):
    rand_mult = 0
    # the while note statement is included because the following implementation has an equal chance to pick -1, 0, and 1
    while not (rand_mult == 1 or rand_mult == -1):
        rand_mult = np.random.randint(-1,2)
    return rand_mult * val
import numpy as np
from scipy.linalg import solve_triangular

def calcularLU(A):

    # Para esta funcion seguimos el algoritmo planteado en el libro Matrix Computations,
    # de Gene H. Golub, Charles F. Van Loan (p. 128)
    L, U = [],[]
    P = None
    
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)
    
    for k in range(n-1):
        max_index = k
        max_valor = abs(U[k,k])
        for i in range(k+1, n):
            if abs(U[i,k]) > max_valor:
                max_valor = abs(U[i,k])
                max_index = i
        
        if max_index != k:
            U[[k,max_index], :] = U[[max_index,k], :]
            P[[k,max_index], :] = P[[max_index,k], :]
            if k > 0:
                L[[k,max_index], :k] = L[[max_index,k], :k]
                
        if U[k,k] != 0:
            for i in range(k+1, n):
                L[i,k] = U[i,k] / U[k,k]
                U[i,k:] = U[i,k:] - L[i,k]*U[k,k:]
    
    

    return L, U, P


def inversaLU(L, U, P=None):
    
    n = L.shape[0]
    I = np.eye(n)
    Inv = np.zeros_like(I)
    
    for i in range(n):
        b = P @ I[:,i]
        y = solve_triangular(L, b, lower=True)
        Inv[:,i] = solve_triangular(U, y, lower=False)
        
    
    return Inv
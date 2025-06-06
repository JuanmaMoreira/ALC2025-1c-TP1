import numpy as np
import scipy.linalg


def calculaLU(A):
    """
    Calcula la factorización LU con pivoteo parcial de la matriz A.
    Retorna matrices L (triangular inferior), U (triangular superior) y P (permutación) tales que:
    PA = LU
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n - 1):
        max_index = k
        max_valor = abs(U[k, k])
        for i in range(k + 1, n):
            if abs(U[i, k]) > max_valor:
                max_valor = abs(U[i, k])
                max_index = i

        if max_index != k:
            U[[k, max_index], :] = U[[max_index, k], :]
            P[[k, max_index], :] = P[[max_index, k], :]
            if k > 0:
                L[[k, max_index], :k] = L[[max_index, k], :k]

        if U[k, k] != 0:
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return L, U, P

def resolver_con_LU(A, b):
    L, U, P = calculaLU(A)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y)
    return x


def power_iteration(A, niter=10_000, eps=1e-6):
    """
    Calcula el autovector y el autovalor asociado más grande en valor absoluto

    Devuelve (a, v) con a autovalor, y v autovector de A

    Arguments:
    ----------

    A: np.array
        Matriz de la cual quiero calcular el autovector y autovalor

    niter: int (> 0)
        Cantidad de iteraciones

    eps: Epsilon
        Tolerancia utilizada en el criterio de parada
    """
    
    # Verificación básica
    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada"
    assert niter > 0, "niter debe ser positivo"
    
    n = A.shape[0]
    v = np.ones(n) / np.linalg.norm(np.ones(n))
    r_prev = 0
    
    for i in range(niter):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
        r_k = (v.T @ A @ v) / (v.T @ v)  # Cociente de Rayleigh
        if (abs(r_k - r_prev) < eps ):
            break
        r_prev = r_k
    

    return r_k, v


def power_iteration_force(A, niter=100):
    """
    Funciona igual que power_iteration pero no tiene criterio de parada y fuerza las niter iteraciones.
    Lo usamos para graficar convergencias hasta niter.
    """
    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada"
    assert niter > 0, "niter debe ser positivo"
    
    n = A.shape[0]
    v = np.ones(n) / np.linalg.norm(np.ones(n))
    r_prev = 0
    approximations = []
    
    for i in range(niter):
        Av = A @ v
        v = Av / np.linalg.norm(Av)
        r_k = (v.T @ A @ v) / (v.T @ v)
        approximations.append(r_k)
        r_prev = r_k

    return r_k, v, approximations


def inverse_power_iteration(A, niter=100, eps=1e-6):
    """
    Calcula el autovector y el autovalor asociado más chico en valor absoluto

    Devuelve (a, v) con a autovalor, y v autovector de A
    """
    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada"
    assert niter > 0, "niter debe ser positivo"
    
    n = A.shape[0]
    v = np.ones(n) / np.linalg.norm(np.ones(n))
    r_prev = 0
    
    for i in range(niter):
        w = resolver_con_LU(A, v)
        v = w / np.linalg.norm(w)
        r_k = v.T @ w
        lambda_k = 1 / r_k if r_k != 0 else np.inf
        if abs(lambda_k - r_prev) < eps:
            break
        r_prev = lambda_k

    return lambda_k, v



def eigen(A, num=2, niter=10000, eps=1e-6):
    """
    Calcula num autovalores y autovectores usando método de la potencia + deflación de Hotelling
    """
    A = A.copy()
    eigenvalues = []
    eigenvectors = np.zeros((A.shape[0], num))
    for i in range(num):
        lambda_k, v_k = power_iteration(A, niter=niter, eps=eps)
        eigenvalues.append(lambda_k)
        
        # Normalizamos el autovector
        v_k = v_k / np.linalg.norm(v_k)
        eigenvectors[:, i] = v_k
        
        # Deflación de Hotelling
        A = A - lambda_k * np.outer(v_k, v_k)
        
    return np.array(eigenvalues), eigenvectors



def eigen2(A, num=2, niter=10000, eps=1e-12):
    """
    FUNCION EXTRA, NO LA VIMOS
    
    Calcula num autovalores y autovectores usando método de la potencia + deflación ortogonal
    La dejo porque eigen da errores en algunos casos de autovalor 0, no creo que la podamos usar en el TP
    """
    A = A.copy()
    eigenvalues = []
    eigenvectors = np.zeros((A.shape[0], num))
    V = []  # Lista para almacenar los autovectores y ortogonalizar
    
    for i in range(num):
        lambda_k, v_k = power_iteration(A, niter=niter, eps=eps)
        eigenvalues.append(lambda_k)
        
        v_k = v_k / np.linalg.norm(v_k)
        
        # Ortogonalizar v_k respecto a los autovectores anteriores
        for v_prev in V:
            v_k = v_k - (v_prev @ v_k) * v_prev
        v_k = v_k / np.linalg.norm(v_k)  # Renormalizar después de ortogonalizar
        
        V.append(v_k)
        eigenvectors[:, i] = v_k
        
        # Proyección ortogonal para la deflación
        P = np.eye(A.shape[0]) - np.outer(v_k, v_k)
        A = P @ A @ P
    
    return np.array(eigenvalues), eigenvectors


# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx


#recibe un grafo, una lista de sus comunidades y una paleta de colores a usar para colorear.
# devuelve el layout adecuado para aristas y vertices segun pertenezcan o no a cada comunidad (colorea segun pertenencia a comunidad)
def graficar_comunidades_en_grafo(G, comunidades, palette="tab20"):
    # 1. Generar colores para cada comunidad
    cmap = plt.get_cmap(palette)
    colores_comunidades = cmap(np.linspace(0, 1, len(comunidades)))
    colores_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                  for r, g, b, _ in colores_comunidades]

    # mapea nodos a colores
    color_por_nodo = {}  # ¡Asegúrate de que sea un diccionario!
    for i, com in enumerate(comunidades):
        for nodo in com:
            color_por_nodo[nodo] = colores_hex[i]  # Nodo → color

    #  colores para aristas (mismo color si misma comunidad)
    edge_colors = []
    for u, v in G.edges():
        if color_por_nodo.get(u) == color_por_nodo.get(v):
            edge_colors.append(color_por_nodo[u])
        else:
            edge_colors.append("#e0e0e0")  # Gris para aristas entre comunidades

    return edge_colors, color_por_nodo  # Devuelve el diccionario correcto



def calcula_K(A):
    # Calcula la matriz de grado K, que tiene en su diagonal la suma por filas de A 
    n = A.shape[0]
    K = np.eye(n)
    for i in range(n):
        suma_fila = np.sum(A[i])
        K[i,i] = suma_fila
    return K


def calcula_L(A):
    K = calcula_K(A)
    L = K - A
    return L

def calcula_R(A):
    K = calcula_K(A)
    k = np.diag(K)
    k_col = k.reshape(-1,1) # vector columna
    k_fil = k.reshape(1,-1) # vector fila
    E = np.sum(A) / 2
    P = (k_col @ k_fil) / (2 * E)
    R = A - P
    return R
    

def calcula_lambda(L,v):
    #Calcula el corte Λ asociado al autovector v usando la matriz de Laplaciano L.
    s = np.sign(v)
    s = s.reshape(-1, 1)  # Vector columna
    return (0.25 * s.T @ L @ s)

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.sign(v)
    s = s.reshape(-1, 1)  # Vector columna
    return (s.T @ R @ s)


def calculaLU(A):
    """
    # Para esta funcion seguimos el algoritmo planteado en el libro Matrix Computations,
    # de Gene H. Golub, Charles F. Van Loan (p. 128)
    Calcula la factorización LU con pivoteo parcial de la matriz A.
    Retorna matrices L (triangular inferior), U (triangular superior) y P (permutación) tales que:
    PA = LU
    """
    n = A.shape[0]  # Obtenemos el tamaño de la matriz A
    # Inicializamos las matrices que vamos a devolver
    L = np.eye(n)   
    U = A.copy()    
    P = np.eye(n)   

    # Recorremos cada columna
    for k in range(n - 1):
        # Paso de pivoteo parcial:
        # Buscamos el índice de la fila con el mayor valor absoluto en la columna k (desde fila k hasta n)
        max_index = k
        max_valor = abs(U[k, k])
        for i in range(k + 1, n):
            if abs(U[i, k]) > max_valor:
                max_valor = abs(U[i, k])
                max_index = i

        # Si el pivote no está en la fila actual, intercambiamos filas en U, P y L
        if max_index != k:
            # Intercambio de filas en U
            U[[k, max_index], :] = U[[max_index, k], :]
            # Intercambio correspondiente en P
            P[[k, max_index], :] = P[[max_index, k], :]
            # Intercambio de las partes ya calculadas de L (hasta la columna k)
            if k > 0:
                L[[k, max_index], :k] = L[[max_index, k], :k]

        # Paso de eliminación gaussiana (solo si el pivote es distinto de 0)
        if U[k, k] != 0:
            for i in range(k + 1, n):
                # Calculamos el multiplicador que anula la entrada debajo del pivote
                L[i, k] = U[i, k] / U[k, k]
                # Eliminamos el elemento actual de la columna
                U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return L, U, P

def resolver_con_LU(A, b):
    # Resuelve el sistema Ax = b usando LU con pivoteo.
    L, U, P = calculaLU(A)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y)
    return x


def metpot1(A, tol=1e-8, maxrep=np.inf):
    """
    Calcula el autovalor de mayor módulo de A y su autovector asociado por el método de la potencia.
    Input: A (matriz cuadrada), tol (tolerancia), maxrep (máximo de iteraciones)
    Retorna: autovector, autovalor, bandera de convergencia
    """
    n = A.shape[0]
    v = np.random.uniform(-1, 1, n)  # Vector aleatorio entre -1 y 1
    v = v / np.linalg.norm(v)  # Normaliza
    l = 0  # Inicializa autovalor estimado
    nrep = 0
    error = tol + 1  # Fuerza al menos una iteración
    while error > tol and nrep < maxrep:
        w = A @ v  # Aplica A
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-10:  # Evita división por cero
            print('Vector nulo detectado')
            return v, l, False
        v_new = w / norm_w  # Normaliza
        l_new = v_new @ (A @ v_new)  # Cociente de Rayleigh
        error = np.abs(l_new - l) / max(np.abs(l_new), 1e-10)  # Error relativo
        v = v_new
        l = l_new
        nrep += 1
    if nrep >= maxrep:
        print('MaxRep alcanzado')
    return v, l, nrep < maxrep

def deflaciona(A, tol=1e-8, maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A, tol, maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1)  # Deflaciona usando producto externo
    return deflA

def metpot2(A, v1, l1, tol=1e-8, maxrep=np.inf):
    # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
    # v1 y l1 son los primeors autovectores y autovalores de A
    deflA = A - l1 * np.outer(v1, v1)
    return metpot1(deflA, tol, maxrep)


def metpotI(A, mu, tol=1e-8, maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu*I, junto a su autovector y si el método convergió.
    n = A.shape[0]
    I = np.eye(n)
    X = A + mu * I  # Calculamos la matriz A shifteada en mu
    # Calcula la inversa de X usando LU
    X_inv = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1  # Vector canónico i-ésimo
        X_inv[:, i] = resolver_con_LU(X, e_i)  # Columna i de la inversa
    return metpot1(X_inv, tol, maxrep)

def metpotI2(A, mu, tol=1e-8, maxrep=np.inf):
    """
    Calcula el segundo autovalor más pequeño de A + mu*I y su autovector usando el método de la potencia inversa con deflación.
    Usa factorización LU para la inversa y aplica metpot1.
    Input: A (matriz cuadrada), mu (escalar de desplazamiento), tol, maxrep
    Retorna: autovector, autovalor (de A + mu*I), bandera de convergencia
    """
    n = A.shape[0]
    I = np.eye(n)
    X = A + mu * I  # Calcula matriz desplazada: A + mu*I
    
    # Calcula la inversa de X usando LU
    iX = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1  # Vector base i-ésimo
        iX[:, i] = resolver_con_LU(X, e_i)  # Columna i de la inversa
    
    # Obtiene el autovalor más grande de iX y su autovector
    v1, l1_inv, _ = metpot1(iX, tol, maxrep)
    v1 = v1 / np.linalg.norm(v1)  # Normalizo
    
    # Deflaciona iX para eliminar el autovalor más grande
    defliX = iX - l1_inv * np.outer(v1, v1)  # Usa producto externo v1 v1^T
    
    # Busca el segundo autovalor más grande de defliX
    v, l_inv, converged = metpot1(defliX, tol, maxrep)
    
    # Convierte autovalor de la inversa al de A + mu*I
    l = 1 / l_inv if abs(l_inv) > 1e-10 else np.inf  # Evita división por cero
    v = v / np.linalg.norm(v)  # Normalizo autovector
    
    return v, l, converged


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    
    np.random.seed(42)  # Fijar semilla para reproducibilidad
    
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
        
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        mu = 0.1
        v,l, _ = metpotI2(L, mu) # Encontramos el segundo autovector de L
        
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        idx_pos = [i for i, vi in enumerate(v) if vi > 0]
        idx_neg = [i for i, vi in enumerate(v) if vi < 0]
        
        Ap = A[np.ix_(idx_pos, idx_pos)] if idx_pos else np.array([[]]) # Asociado al signo positivo
        Am = A[np.ix_(idx_neg, idx_neg)] if idx_neg else np.array([[]]) # Asociado al signo negativo
        
        # Nombres correspondientes a cada grupo
        nombres_pos = [nombres_s[i] for i in idx_pos]
        nombres_neg = [nombres_s[i] for i in idx_neg]
        
        return (
            laplaciano_iterativo(Ap, niveles - 1, nombres_pos) +
            laplaciano_iterativo(Am, niveles - 1, nombres_neg)
        )       


def modularidad_iterativo(A=None, R=None, nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    np.random.seed(42)  # Fijar semilla para reproducibilidad
    
    if A is None and R is None:
        print('Dame una matriz')
        return np.nan
    
    if R is None:
        R = calcula_R(A)
    
    if nombres_s is None:
        nombres_s = list(range(R.shape[0]))
    
    # Acá empieza lo bueno
    if R.shape[0] == 1:  # Si llegamos al último nivel
        return [nombres_s]
    
    else:
        # Primer autovector y autovalor de R
        v, l, _ = metpot1(R)
        
        # Modularidad actual: Q0 = s^T R s (usando calcula_Q)
        Q0 = float(calcula_Q(R, v)) / (2 * np.sum(A) / 2) if A is not None and np.sum(A) > 0 else 0
        
        if Q0 <= 0 or all(v > 0) or all(v < 0):  # Si la modularidad es no positiva o no hay partición
            return [nombres_s]
        
        else:
            # Submatrices de R asociadas a los valores positivos y negativos de v
            idx_pos = [i for i, vi in enumerate(v) if vi > 0]
            idx_neg = [i for i, vi in enumerate(v) if vi < 0]
            Rp = R[np.ix_(idx_pos, idx_pos)] if idx_pos else np.array([[]])
            Rm = R[np.ix_(idx_neg, idx_neg)] if idx_neg else np.array([[]])
            
            # Autovectores principales de Rp y Rm
            vp, lp, _ = metpot1(Rp) if idx_pos else (np.array([]), 0, True)
            vm, lm, _ = metpot1(Rm) if idx_neg else (np.array([]), 0, True)
            
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if len(idx_pos) > 1 and not (all(vp > 0) or all(vp < 0)):
                Q1 += float(calcula_Q(Rp, vp)) / (2 * np.sum(A) / 2) if A is not None and np.sum(A) > 0 else 0
            if len(idx_neg) > 1 and not (all(vm > 0) or all(vm < 0)):
                Q1 += float(calcula_Q(Rm, vm)) / (2 * np.sum(A) / 2) if A is not None and np.sum(A) > 0 else 0
            
            if Q0 >= Q1:  # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return [
                    [nombres_s[i] for i in idx_pos],
                    [nombres_s[i] for i in idx_neg]
                ]
            else:
                # Sino, repetimos para los subniveles
                return (
                    modularidad_iterativo(A, Rp, [nombres_s[i] for i in idx_pos]) +
                    modularidad_iterativo(A, Rm, [nombres_s[i] for i in idx_neg])
                )


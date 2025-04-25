import numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx


def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(A):
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

def resolver_con_LU(A, b):
    # Resuelve el sistema Ax = b usando LU con pivoteo.
    L, U, P = calculaLU(A)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y)
    return x

def calcular_matriz_K_inv(A):
    # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A 
    dimFilas = A.shape[0]
    Kinv = np.eye(dimFilas) #esto esta ok si la matriz es cuadrada.
    for i in range(dimFilas):
        suma_fila = np.sum(A[i])
        Kinv[i,i] = 0 if suma_fila == 0 else 1 / suma_fila
    return Kinv

def calcular_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C 
    #Construyo la matriz K inversa directamente
    Kinv = calcular_matriz_K_inv(A)    
    return np.transpose(A) @ Kinv # Calculo C
 
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcular_matriz_C(A)
    N = A.shape[0]  # Número de museos
    I = np.eye(N)

    M = (N / alfa) * (I - (1 - alfa) * C)
    b = np.ones(N)
    p = resolver_con_LU(M, b)
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = calcular_matriz_K_inv(F) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = np.transpose(F) @ Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    C_k = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        C_k = C_k @ C
        B += C_k # Sumamos las matrices de transición para cada cantidad de pasos
    
    return B

def norma_1_matricial(A):
    # Suma máxima por columnas
    return np.max(np.sum(np.abs(A), axis=0))

def condicion_1_por_LU(A):
    # Devuelve el número de condición 1 de la matriz A
    n = A.shape[0]
    Ainv = np.zeros_like(A, dtype=float)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        Ainv[:, i] = resolver_con_LU(A, e)

    return norma_1_matricial(A) * norma_1_matricial(Ainv)
def graficar_red_museos(G, G_layout, barrios, p, titulo, tamaño_base=75000, ax=None):
    """
    Grafica la red de museos sobre el mapa, asignando tamaños de nodo proporcionales a PageRank.

    Parámetros:
    ------------
    G : networkx.Graph
        Grafo de museos.
    G_layout : dict
        Layout de coordenadas (posición de cada museo).
    barrios : geopandas.GeoDataFrame
        Geometría de los barrios de CABA.
    p : numpy.ndarray
        Vector de PageRank de los museos.
    titulo : str
        Título del gráfico.
    tamaño_base : float
        Constante para escalar los tamaños de los nodos.
    ax : matplotlib.axes.Axes, opcional
        Eje en el que dibujar el gráfico. Si no se da, se crea uno nuevo.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Graficar barrios
    barrios.to_crs("EPSG:22184").boundary.plot(color='#bbbbbb', ax=ax)

    # Calcular tamaño de nodos
    node_sizes = tamaño_base * p

    # Dibujar red
    nx.draw_networkx(
        G, pos=G_layout, ax=ax,
        node_size=node_sizes,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
        width=1.5,
        with_labels=False
    )

    ax.set_title(titulo, fontsize=12)
    ax.axis('off')





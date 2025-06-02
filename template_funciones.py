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

def calcular_matriz_K_inv(A):
    # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A 
    dimFilas = A.shape[0]
    Kinv = np.eye(dimFilas)
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
    with np.errstate(divide='ignore'):  # evita el warning de división por cero
        F = 1 / D
    np.fill_diagonal(F,0) # Reemplaza los inf de la diagonal por ceros
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

def graficar_redes_museos_set(grafos, pageranks, G_layout, barrios, titulo_general, tamaño_base=75000, n_cols=2):
    """
    Grafica un set de redes de museos con sus respectivos pageranks en una grilla.

    Parámetros:
    -----------
    grafos : dict
        Diccionario {clave: grafo}, donde la clave puede ser el valor de m o alfa.
    pageranks : dict
        Diccionario {clave: vector de pagerank}, con las mismas claves que 'grafos'.
    G_layout : dict
        Diccionario de posiciones de los nodos.
    barrios : geopandas.GeoDataFrame
        Geometría de los barrios de CABA.
    titulo_general : str
        Título general para todo el set de gráficos.
    tamaño_base : float
        Escalado base para el tamaño de los nodos.
    n_cols : int
        Cantidad de columnas en la grilla.
    """
    claves = list(grafos.keys())
    n = len(claves)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    axes = axes.flatten()

    for i, clave in enumerate(claves):
        ax = axes[i]
        G = grafos[clave]
        p = pageranks[clave]

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

        ax.set_title(f"{titulo_general}: {clave}", fontsize=12)
        ax.axis('off')

    # Borrar ejes sobrantes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(titulo_general, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()





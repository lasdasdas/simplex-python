import numpy as np
import itertools


def combinations(A, B, names):
    m = A.shape[0]
    n = A.shape[1]
    # Todas las combinaciones posibles de columnas
    indices = [i for i in range(n)]
    comb = list(itertools.combinations(indices, 3))
    ret = []
    # Por cada una de las combinaciones
    for c in comb:
        tmp = np.zeros((m, m))
        var_list = []
        for i in range(m):
            # Extraemos la columna globlar (como fila)
            tmp[i, :] = np.array(A[:, c[i]])
            # Guardamos el nombre también
            var_list.append(names[c[i]])
        ret.append({"arr": tmp.transpose().copy(),
                    "var": var_list,
                    "J": list(set(indices) - set(c)),
                    "I": list(c)})
    return ret


def solve(comb_list, B):
    for ele in comb_list:
        sol = np.matmul(np.linalg.inv(ele["arr"]), B.transpose())
        ele["sol"] = sol


def calc_cost(comb_list, cost_dic):
    for comb in comb_list:
        cost = 0.0
        for var, res in zip(comb["var"], comb["sol"].tolist()):
            if var in cost_dic:
                cost += cost_dic[var] * res
        comb["cost"] = cost


def is_viable(comb_list):
    ret = []
    for c in comb_list:
        if np.all(c["sol"] > 0):
            ret.append(c)
    return ret


# Definimos la matrix fundamental como una matrix numpy
A = np.array([[-1., 2, 1, 0, 0],
              [2,  3, 0, 1, 0],
              [4, -4, 0, 0, 1],
              ])
# Definimos términos independientes
B = np.array([4, 12, 12])

# Nombres de las variables para mejor identificación
names = ["x1", "x2", "h1", "h2", "h3"]


# Función de coste en formato diccionario
cost = {"x1": 4, "x2": 1}
C = np.array([4, 1, 0, 0, 0])

# Contiene todas las submatrices
comb = (combinations(A, B, names))


# Calculamos las soluciones del sistema y las añadimos al diccionario
solve(comb, B)
calc_cost(comb, cost)

# Filramos las no viables
sv = is_viable(comb)


# Apartado 3
def simplex(A, B, C):
    m = A.shape[0]
    n = A.shape[1]

    Biter = B.copy()

    # Matrix A y Ap, la subbase
    A = np.matrix(A)
    Ap = np.matrix(A[:, -m:])

    # Columnas que usare en la subbase
    cols = [i for i in range(n-m, n)]

    # Columnas que no usare en la subbase
    no_cols = [i for i in range(0, m-1)]
    z = 0
    solution = []

    # Iteramos hasta converger
    for iteration in itertools.count(0):
        # Inversa de la subbase
        Apinv = np.linalg.inv(Ap)

        # Elementos de las columnas que estamos usando de la matriz de coste
        cb = np.array([C[i] for i in cols])

        # Coeficiente absoluto de z-c(vector)
        z_minus_c = np.zeros((n-m))

        for c_i, i in zip(no_cols, range(n-m)):
            z_minus_c[i] = float(np.matmul(cb,
                                 np.matmul(Apinv, A[:, c_i]))-C[c_i])
            max_col_arg = np.ma.argmin(z_minus_c, fill_value=[z_minus_c < 0])
        max_col = no_cols[max_col_arg]

        xsk = np.matmul(Apinv, B)
        ysk = np.matmul(Apinv, A[:, max_col]).transpose()
        Biter = np.multiply(np.divide(xsk, ysk), (ysk > 0).astype(int))
        Biter[Biter == 0] = np.inf

        z = float(cb * xsk.transpose())
        print("Iter "+str(iteration) + " ,cost is " + str(z))

        min_col_arg = np.argmin(Biter[np.nonzero(Biter)])
        min_col = cols[min_col_arg]

        if np.all(z_minus_c > 0):
            print("Exit case 1")
            solution = xsk.flatten().tolist()[0]
            break
        if np.all(ysk < 0):
            print("Exit case 2")
            solution = xsk.tolist()
            break
        no_cols[max_col_arg] = min_col
        cols[min_col_arg] = max_col
        Ap[:, min_col_arg] = A[:, max_col]

    print("Ap is", Ap)
    print("no cols is", no_cols)
    print("cols is   ", cols)
    print("solution is:")
    for i in range(m):
        print("variable in column: " + str(cols[i]) +
              " -> " + str(solution[i]))

    return solution, cols, Ap


A = np.array([[-1., 2, 1, 0, 0],
              [2,  3, 0, 1, 0],
              [4, -4, 0, 0, 1],
              ])
A = np.reshape(A, (3, 5))
B = np.array([4, 12, 12])
C = np.array([4, 1, 0, 0, 0])
simplex(A, B, C)

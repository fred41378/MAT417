import numpy as np

A = np.array(
    [[0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]])

b = np.array([0.1, 0.3, 0.5])

def gauss_simple(A, b):
    n = len(b)
    # M est une copie de A qui va devenir une matrice triangulaire supperieur
    M = np.array(A, dtype=np.float32)
    # v est une copie de b qui va devenir une solution en meme temps que M va evoluer
    v = np.array(b, dtype=np.float32)

    # On applique l'algo des notes de cours pour chaque etapes
    for k in range(n):
        for i in range(k+1, n):
            facteur_m = M[i, k] / M[k, k]
            M[i, k:] -= facteur_m * M[k, k:]
            v[i]     -= facteur_m * v[k]

    # Substitution arrière
    x = np.zeros(n, dtype=np.float32)
    for i in range(n-1, -1, -1):
        x[i] = (v[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]

    return x

def gauss_double(A, b):
    n = len(b)
    # M est une copie de A qui va devenir une matrice triangulaire supperieur
    M = np.array(A, dtype=np.float64)
    # v est une copie de b qui va devenir une solution en meme temps que M va evoluer
    v = np.array(b, dtype=np.float64)

    # On applique l'algo des notes de cours pour chaque etapes
    for k in range(n):
        for i in range(k+1, n):
            facteur_m = M[i, k] / M[k, k]
            M[i, k:] -= facteur_m * M[k, k:]
            v[i]     -= facteur_m * v[k]

    # Substitution arrière
    x = np.zeros(n, dtype=np.float64)
    for i in range(n-1, -1, -1):
        x[i] = (v[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]

    return x

def GaussElimination(a, b, simplePrecision):

    if simplePrecision is True:
        mat = np.float32(a)
        vec = np.float32(b)
    else:
        mat = a
        vec = b

    # Élimination
    for i in range(len(mat) - 1):  # Ligne avec le pivot
        pivot = mat[i][i]
        for j in range(i + 1, len(mat)): # Lignes en dessous
            m = mat[j][i]/pivot
            for k in range(len(mat[j])):
                mat[j][k] = mat[j][k] - m * mat[i][k]

            vec[j] = vec[j] - m * vec[i]

    # Substituion arrière
    sol = [0] * len(vec)
    for l in range(len(mat) - 1, -1, -1):
        sol[l] = (vec[l] - sum(mat[l][j] * sol[j] for j in range(l + 1, len(mat)))) / mat[l][l]

    return mat, vec, sol

solution_moi_simple = gauss_simple(A,b)
solution_moi_double = gauss_double(A,b)
_, _, solution_ana = GaussElimination(A,b,True)

print("solution_moi_simple :\n", solution_moi_simple)
print("solution_moi_double :\n", solution_moi_double)
print("solution_ana :\n", solution_ana)
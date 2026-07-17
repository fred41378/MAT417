import numpy as np

A = np.array(
    [[21.0, 67.0, 88.0, 73.0],
     [76.0, 63.0, 7.0, 20.0],
     [0.0, 85.0, 56.0, 54.0],
     [19.3, 43.0, 30.2, 29.4]]).astype(np.float32)

b = np.array([141.0, 109.0, 218.0, 93.7]).astype(np.float32)

print("A:\n", A)
print("b:\n", b)

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

    return M, v, x.astype(np.float32)


mat_a,vec_a,x = gauss_simple(A, b)

print("A echelon \n",mat_a)
print("b echelon \n",vec_a)

print("Question a)\n solution par Gauss en simple precision : \n", x)
A_copie = A.copy()
r = (b - A_copie @ x).astype(np.float32)
print("Question b)\n residu r : \n", r)


_,_,z = gauss_simple(A, r)
x += z
print("Qestion c)\n z est : \n", z)
print("\n nouveau x est : \n", x)

print("Question d)\n repetition des etapes jusqu'a ce qu'aucune amelioration soit observe : \n", z)
for n in range(10):
    r = (b - A_copie @ x).astype(np.float32)
    _,_,z = gauss_simple(A, r)
    x += z
    print("x",n+1," : ", x)

print("On constate que x arrete de s'ameliorer a x7 \n")


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

x_double = gauss_double(A, b)
print("Question d) \n Resolution en double precision : \n", x_double)

print("On constate qu'en double precision, la solution est bien plus exacte")
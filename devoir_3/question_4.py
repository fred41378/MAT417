import numpy as np
import matplotlib.pyplot as plt


A = np.array([
    [7,  1, -1,  2],
    [1,  8,  0, -2],
    [-1, 0,  4, -1],
    [2, -2, -1,  6]
])

b = np.array([3, -5, 4, -3])

# Solution exacte
x_exacte = np.array([1, -1, 1, -1])

tol = 1e-5

def jacobi(A, b, x0, x_exact):
    n = len(b)
    x = x0.copy()

    errors = [np.linalg.norm(x - x_exact)]

    it = 0
    x_new = np.zeros(n)

    while np.linalg.norm(x - x_exact) > tol:
        x_new = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]

        errors.append(np.linalg.norm(x_new - x_exact))

        x = x_new
        it += 1

    return x_new, it + 1, errors


def gauss_seidel(A, b, x0, x_exact):
    n = len(b)
    x = x0.copy()

    errors = [np.linalg.norm(x - x_exact)]
    it = 0
    while np.linalg.norm(x - x_exact) > tol:
        x_old = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        errors.append(np.linalg.norm(x - x_exact))
        it += 1

    return x, it + 1, errors


def sor(A, b, x0, x_exact):
    n = len(b)
    x = x0.copy()
    w = 1.12

    errors = [np.linalg.norm(x - x_exact)]
    it = 0
    while np.linalg.norm(x - x_exact) > tol:
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])

            x[i] = ((1 - w) * x_old[i] + w * (b[i] - s1 - s2) / A[i, i])

        errors.append(np.linalg.norm(x - x_exact))
        it += 1

    return x, it + 1, errors


# Point initial
x0 = np.zeros(4)

# Résolution
x_jacobi, it_jacobi, err_jacobi = jacobi(A, b, x0, x_exacte)

x_gs, it_gs, err_gs = gauss_seidel(A, b, x0, x_exacte)

x_sor, it_sor, err_sor = sor(A, b, x0, x_exacte)


print("Jacobi :")
print("Solution :", x_jacobi)
print("Itérations :", it_jacobi)
print()

print("Gauss-Seidel :")
print("Solution :", x_gs)
print("Itérations :", it_gs)
print()

print("SOR :")
print("Solution :", x_sor)
print("Itérations :", it_sor)
print()

# Affichage des erreurs
plt.figure(figsize=(8, 5))

plt.plot(np.log(err_jacobi), label="Jacobi")
plt.plot(np.log(err_gs), label="Gauss-Seidel")
plt.plot(np.log(err_sor), label=f"SOR (w=1.12)")

plt.xlabel("Itération k")
plt.ylabel("||x^(k) - x_exact||")
plt.title("Comparaison des méthodes")
plt.grid(True)
plt.legend()

plt.show()
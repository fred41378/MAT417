import numpy as np
import matplotlib.pyplot as plt
import time

# Initialisation

def getErrors(n):
    numbers = np.random.rand(n)
    times = []

    # Méthode I
    sum_I = np.double(0.0)
    for num in numbers:
        sum_I = sum_I + num

    # Méthode II
    t0 = time.perf_counter()
    sum_II = np.single(0.0)
    for num in numbers:
        sum_II = np.single(sum_II + np.single(num))
    times.append(time.perf_counter() - t0)

    # Méthode III
    t0 = time.perf_counter()
    sum_III = np.single(numbers[0])
    c = np.single(0.0)
    for i in range (1, n):
        y = np.single(numbers[i] - c)
        t = np.single(sum_III + y)
        c = np.single((t - sum_III) - y)
        sum_III = t
    times.append(time.perf_counter() - t0)

    # Méthode IV
    t0 = time.perf_counter()
    sorted_numbers = np.sort(numbers)
    sum_IV = np.single(0.0)
    for s_num in sorted_numbers:
        sum_IV = np.single(sum_IV + np.single(s_num))
    times.append(time.perf_counter() - t0)

    # Méthode V
    t0 = time.perf_counter()
    sum_V = np.single(0.0)
    for r_num in reversed(sorted_numbers):
        sum_V = np.single(sum_V + np.single(r_num))
    times.append(time.perf_counter() - t0)

    # Erreurs absolues

    results = [sum_II, sum_III, sum_IV, sum_V]

    return [np.abs(sum_I - r)/np.abs(sum_I) for r in results], times

n = np.arange(10000, 500000, 10000)

# 1. Création du graphique

data = np.zeros(shape=(len(n), 4))
data_times = np.zeros(shape=(len(n), 4))

for i in range(len(n)):
    data[i], data_times[i] = getErrors(n[i])

plt.plot(n, data[:, 0], label='Méthode II')
plt.plot(n, data[:, 1], label='Méthode III')
plt.plot(n, data[:, 2], label='Méthode IV')
plt.plot(n, data[:, 3], label='Méthode V')
plt.title("Erreurs absolues en fonction de n")
plt.xlabel("n")
plt.ylabel("Erreurs absolues")
plt.legend()
plt.show()

# 2. Erreurs moyennes

# Méthode II
avg_err_II = np.mean(data[:, 0])
print(avg_err_II)

# Méthode III
avg_err_III = np.mean(data[:, 1])
print(avg_err_III)

# Méthode IV
avg_err_IV = np.mean(data[:, 2])
print(avg_err_IV)

# Méthode V
avg_err_V = np.mean(data[:, 3])
print(avg_err_V)

# La méthode III est celle qui donne la plus petite erreur relative.
# La méthode IV et V produisent une erreur relative très similaire,
# soit la plus grande erreur relative moyenne.
# La méthode II produit une erreur relative plus grande que III, mais
# tout de même mieux que IV et V.

# 3.

logN = np.log(n)

# Méthode II
logT_II = np.log(data_times[:, 0])
k_II, lnC_II = np.polyfit(logN, logT_II, 1)

# Méthode III
logT_III = np.log(data_times[:, 1])
k_III, lnC_III = np.polyfit(logN, logT_III, 1)

# Méthode IV
logT_IV = np.log(data_times[:, 2])
k_IV, lnC_IV = np.polyfit(logN, logT_IV, 1)

# Méthode V
logT_V = np.log(data_times[:, 3])
k_V, lnC_V = np.polyfit(logN, logT_V, 1)

plt.plot(logN, logT_II, label=f'Méthode II (k={k_II:.3f})')
plt.plot(logN, logT_III, label=f'Méthode III (k={k_III:.3f})')
plt.plot(logN, logT_IV, label=f'Méthode IV (k={k_IV:.3f})')
plt.plot(logN, logT_V, label=f'Méthode V (k={k_V:.3f})')
plt.title("Temps d'execution en fonction de n")
plt.xlabel("ln(n)")
plt.ylabel("Temps d'execution")
plt.legend()
plt.show()

# Comparaisons

# On voit facilement que la méthode III a une plus grande ordonnée
# à l'origine donc une plus grande constante C.
# Cependant les 4 méthodes ont une pente relativement similaire.

# 4. L'algorithme de Kahan fonctionne bien car il utilise une variable
# c pour conserver les nombres qui seraient perdus lors d'une addition normale.
# À la prochaine itération, on ré-utilise les petits nombres qui ont
# été perdus précédemment et on les additionne.
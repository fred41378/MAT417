import matplotlib.pyplot as plt
import numpy as np
import time

n0 = 10000
n1 = 500000
n = np.arange(start=n0, stop= n1 + 10000, step=10000)

err2 = []
err3 = []
err4 = []
err5 = []

times2 = []
times3 = []
times4 = []
times5 = []


for i in n:
    x = np.random.default_rng().random(i, dtype=np.float32)
    methode1 = sum(np.float64(x))

    start_time = time.perf_counter()
    methode2 = sum(np.float32(x))
    times2.append(max(time.perf_counter() - start_time, 1e-12))

    s = x[0]
    c = 0
    start_time = time.perf_counter()
    for j in range(1, i):
        y = x[j] - c
        t = s + y
        c = (t - s) - y
        s = t
    methode3 = np.float32(s)
    times3.append(max(time.perf_counter() - start_time, 1e-12))

    start_time = time.perf_counter()
    x_sorted_asc = np.float32(np.sort(x))
    methode4 = sum(x_sorted_asc)
    times4.append(max(time.perf_counter() - start_time, 1e-12))

    start_time = time.perf_counter()
    x_sorted_desc = np.float32(np.sort(x[::-1]))
    methode5 = sum(x_sorted_desc)
    times5.append(max(time.perf_counter() - start_time, 1e-12))

    err2.append(abs(methode2 - methode1))
    err3.append(abs(methode3 - methode1))
    err4.append(abs(methode4 - methode1))
    err5.append(abs(methode5 - methode1))



plt.plot(n, err2, color='orange', label='err2', linestyle='dashdot')
plt.plot(n, err3, color='green', label='err3', linestyle='solid')
plt.plot(n, err4, color='red', label='err4', linestyle='solid')
plt.plot(n, err5, color='blue', label='err5', linestyle='dotted')
plt.legend()
plt.title('Devoir 1')
plt.xlabel('n')
plt.ylabel('Error')
plt.show()

meanerr2 = np.mean(err2)
meanerr3 = np.mean(err3)
meanerr4 = np.mean(err4)
meanerr5 = np.mean(err5)

print("Classement des methodes : \n")
print("4e: methode4 (" + str(meanerr4) +")")
print("4e: methode5 (" + str(meanerr5) + ")")
print("2e: methode2 (" + str(meanerr2) + ")")
print("1e: methode3 (" + str(meanerr3) + ")")
print("On peut voir que les méthodes qui orde en ordre croissant ou décroissant sont égales \n ",
      "et que la meilleur méthode est de loins la 3e")

log_n = np.log(n)
log_times2 = np.log(times2)
log_times3 = np.log(times3)
log_times4 = np.log(times4)
log_times5 = np.log(times5)

k2, a2 = np.polyfit(log_n, log_times2, 1)
k3, a3 = np.polyfit(log_n, log_times3, 1)
k4, a4 = np.polyfit(log_n, log_times4, 1)
k5, a5 = np.polyfit(log_n, log_times5, 1)

reg2 = a2 + k2 * log_n
reg3 = a3 + k3 * log_n
reg4 = a4 + k4 * log_n
reg5 = a5 + k5 * log_n

plt.plot(log_n, log_times2, 'o', color='orange',  label='methode 2')
plt.plot(log_n, log_times3, 'o', color='green',  label='methode 3')
plt.plot(log_n, log_times4, 'o', color='red',  label='methode 4')
plt.plot(log_n, log_times5, 'o', color='blue',  label='methode 5')

plt.plot(log_n, reg2, color='orange', label='regression 2')
plt.plot(log_n, reg3, color='green', label='regression 3')
plt.plot(log_n, reg4, color='red', label='regression 4')
plt.plot(log_n, reg5, color='blue', label='regression 5')

plt.xlabel("ln(n)")
plt.ylabel("ln(T(n))")
plt.title("Régression log-log")
plt.legend()

plt.show()

print("On peut voir avec le deuxième grapgiques que la raison pourquoi la méthode 3 est beaucoup plus efficace, c'est parce qu'elle est beaucoup plus lourde en terme de calcul. ")
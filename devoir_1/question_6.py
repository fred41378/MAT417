import matplotlib.pyplot as plt
import random

n = 10

x = [random.random() for i in range(n)]

print(x)
plt.plot(x)
plt.show()
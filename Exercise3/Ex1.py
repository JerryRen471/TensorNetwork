from math import exp
import numpy as np
import matplotlib.pyplot as plt

def calc_Z(T):
    ans = 0
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                for s4 in [-1, 1]:
                    ans += exp(-(s1*s2 + s2*s3 + s3*s4 + s4*s1)/T)
    return ans
tempratures = np.linspace(0.1, 5)
Zs = np.zeros(len(tempratures))
for i in range(len(tempratures)):
    T = tempratures[i]
    Zs[i] = calc_Z(T)

hand = (exp(-1) + exp(1))**4 + (exp(-1) - exp(1))**4
print(hand)

plt.plot(tempratures, Zs)
plt.show()
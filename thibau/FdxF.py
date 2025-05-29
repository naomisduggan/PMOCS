import numpy as np
import matplotlib.pyplot as plt

# Parameter set
b = 1.0
K = 1.0
n = 5
kx=1.7

a0 = 0.1
b0 = 1.0
K0 = 1.0
m = 5

a_val = 0.3

def f(x, a=a_val):
    return a + b * x**n / (K**n + x**n)

def g(x):
    return a0 + b0 * K0**m / (K0**m + x**m)

def df_dx(x, a=a_val):
    return b * n * K**n * x**(n-1) / (K**n + x**n)**2

def dg_dx(x):
    return -b0 * m * K0**m * x**(m-1) / (K0**m + x**m)**2

def dF_dx(x, y, a=a_val):
    return -f(x, a) + (y - x) * df_dx(x, a) - g(x) - x * dg_dx(x)

def secdefF(x,y,a=a_val,h=1e-8):
    return (dF_dx(x+h,y,a)-dF_dx(x-h,y,a))/(2*h)

def F(x, y, a=a_val):
    return (y - x)*f(x, a) - x * g(x)

def G(x,y):
    return kx-x*y

x_vals = np.linspace(0, 3, 400)
y_vals = np.linspace(0, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
dF = dF_dx(X, Y)
F_vals = F(X, Y)
ddF = secdefF(X,Y)

plt.figure(figsize=(6, 6))
cp = plt.contourf(Y, X, dF, levels=30)
cs1 = plt.contour(Y, X, dF, levels=[0], colors='black', linewidths=2)
plt.clabel(cs1, fmt='∂F/∂x=0')
cs2 = plt.contour(Y, X, F_vals, levels=[0], colors='white', linestyles='--', linewidths=2)
plt.clabel(cs2, fmt='F=0')
#plt.title(f"$\partial_x F$ and critical manifold $F=0$ (a ={a_val}, $k_x$ ={kx})")
plt.xlabel('$X_T$')
plt.ylabel('$X$')
plt.colorbar(cp, label=r'$\partial_X F$')
plt.tight_layout()
plt.show()

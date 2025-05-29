import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ---------- parameters ----------
p = dict(b=1.0, K=1.0, n=5, ap=0.1, bp=1.0, Kp=1.0, m=5)
delta    = 0.1
Delta_a  = 0.3
k_switch = 4.0
Xc       = 1.0

# your fold finder
def f(x, a):
    return a + p['b']*x**p['n']/(p['K']**p['n']+x**p['n'])
def g(x):
    return p['ap'] + p['bp']*p['Kp']**p['m']/(p['Kp']**p['m']+x**p['m'])
def H(y, abar):
    return abar + Delta_a*np.tanh(k_switch*(y-Xc))

def df_dx(x, a, h=1e-8):
    return (f(x+h, a)-f(x-h, a))/(2*h)
def dg_dx(x, h=1e-8):
    return (g(x+h)-g(x-h))/(2*h)

def F_eq(xy, a):
    x,y = xy
    return (y-x)*f(x,a) - x*g(x)
def dFdx_eq(xy, a):
    x,y = xy
    return -f(x,a) + (y-x)*df_dx(x,a) - g(x) - x*dg_dx(x)

def find_folds_for_a(a_val):
    guesses = [(0.7,0.1),(1.3,1.7)]
    folds=[]
    for guess in guesses:
        sol,_,ier,_ = fsolve(lambda v: [F_eq(v,a_val),dFdx_eq(v,a_val)],
                             guess, full_output=True)
        if ier==1:
            folds.append(sol)
    folds = np.array(folds)
    # sort by x so folds[0]=left knee, folds[1]=right knee
    idx = np.argsort(folds[:,0])
    return folds[idx]

# analytic drift D(x,a;kX,abar)
def drift_D(x, a, kX, abar):
    fval = f(x,a)
    gval = g(x)
    y    = x*(1 + gval/fval)
    term1= delta*fval*(kX - x**2)
    term2= -delta*gval*x**2
    term3= (gval/fval)*x*(H(y,abar)-a)
    return term1 + term2 + term3

# parameter grid
kX_grid   = np.linspace(0.3,2.3,200)
abar_grid = np.linspace(0.05,1,200)
KX, ABAR  = np.meshgrid(kX_grid, abar_grid)

Dleft  = np.full_like(KX, np.nan)
Dright = np.full_like(KX, np.nan)

# sweep
for i, abar in enumerate(abar_grid):
    for j, kX in enumerate(kX_grid):
        try:
            x1,y1 = find_folds_for_a(abar)[0]
            x2,y2 = find_folds_for_a(abar)[1]
            a1 = H(y1,abar) 
            a2 = H(y2,abar)
            Dleft [i,j] = drift_D(x1, a1, kX, abar)
            Dright[i,j] = drift_D(x2, a1, kX, abar)
        except Exception:
            pass

# plot
plt.figure(figsize=(5,5))
csL = plt.contour(KX, ABAR, Dleft , levels=[0], colors='blue',  linewidths=2)
csR = plt.contour(KX, ABAR, Dright, levels=[0], colors='red',   linewidths=2)
plt.clabel(csL, fmt='$D_{left}=0$', inline = True)
plt.clabel(csR, fmt='$D_{right}=0$', inline = True)
plt.xlabel(r'$k_X$')
plt.ylabel(r'$\bar a$')
#plt.title(f'Bifurcation curves $D=0$ for $\delta$ = {delta} and $\Delta a$ = {Delta_a}')
plt.tight_layout()
plt.show()

# GSPT for full dynamics
# Also looks at limit cycle in a (full drift)
# If the drift plots end up on the other side of the 0-axis you get a limit cycle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ---------- parameters ----------
p = dict(b=1.0, K=1.0, n=5, ap=0.1, bp=1.0, Kp=1.0, m=5)
kX    = 1.7     # production rate in ẏ, seems to shift both drifts, 1.7 is default
delta = .005      # timescale of ȧ choose something in the range (delta 0.1 --> 1), doesn't seem to matter too much for this

# parameters for H(y)
abar      = 0.2
Delta_a   = 0.5
k_switch  = 4.0
Xc        = 1.0

# Hill functions
def f(x, a):
    return a + p['b'] * x**p['n'] / (p['K']**p['n'] + x**p['n'])

def g(x):
    return p['ap'] + p['bp'] * p['Kp']**p['m'] / (p['Kp']**p['m'] + x**p['m'])

# finite-difference derivatives

def df_dx(x,a,h=1e-8):
    return (f(x+h,a) - f(x-h,a)) / (2*h)

def dg_dx(x, h=1e-8):
    return (g(x+h) - g(x-h)) / (2*h)

# slow a-dynamics
def H(y):
    return abar + Delta_a * np.tanh(k_switch * (y - Xc))

# Fast subsystem functions
def F_eq(xy, a):
    x, y = xy
    return (y - x) * f(x, a) - x * g(x)

def dFdx_eq(xy, a):
    x, y = xy
    return -f(x, a) + (y - x) * df_dx(x, a) - g(x) - x * dg_dx(x)

# Find folds for a given 'a'
def find_folds_for_a(a_val):
    def system(vars):
        return [F_eq(vars, a_val), dFdx_eq(vars, a_val)]
    guesses = [(0.5,0.2), (1.7, 1.5)] 
    #play aorund with these to find good plots, sometimes the wrong folds seem to be picked, for the parameters now, this is perfect
    folds = []
    for guess in guesses:
        sol, _, ier, _ = fsolve(system, guess, full_output=True)
        if ier == 1:
            folds.append(sol)
    return np.array(folds)

# Sweep 'a' values
a_grid = np.linspace(0.1, 0.5, 100)
folds1 = np.zeros((len(a_grid), 2))
folds2 = np.zeros((len(a_grid), 2))
Dmin1  = np.zeros(len(a_grid))
Dmin2  = np.zeros(len(a_grid))
Dfull1 = np.zeros(len(a_grid))
Dfull2 = np.zeros(len(a_grid))

for i, a_val in enumerate(a_grid):
    folds = find_folds_for_a(a_val)
    x1, y1 = folds[0]
    x2, y2 = folds[1]
    folds1[i] = [x1, y1]
    folds2[i] = [x2, y2]
    # minimal drifts, only limit cycle in x,y plane predicted
    Dmin1[i] = kX - x1 * y1
    Dmin2[i] = kX - x2 * y2
    # full drifts, limit cycle in x,y and a (full dynamics)
    Dfull1[i] = f(x1, a_val)*(kX - x1*y1) + (y1 - x1)*(H(y1) - a_val)/delta
    Dfull2[i] = f(x2, a_val)*(kX - x2*y2) + (y2 - x2)*(H(y2) - a_val)/delta

# Plot fold points in (x,y) colored by 'a'
plt.figure(figsize=(6, 6))
plt.scatter(folds1[:,0], folds1[:,1], c=a_grid, cmap='viridis', s=30, label='Fold 1')
plt.scatter(folds2[:,0], folds2[:,1], c=a_grid, cmap='viridis', s=30, marker='x', label='Fold 2')
plt.colorbar(label='a value')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Fold Points in (x, y) colored by a for kx ={kX}')
plt.legend()
plt.tight_layout()
plt.show()

# Plot minimal drifts vs a
plt.figure(figsize=(6, 4))
plt.plot(a_grid, Dmin1, 'o-', label='D_min at Fold 1')
plt.plot(a_grid, Dmin2, 'x-', label='D_min at Fold 2')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('a')
plt.ylabel('D_min')
plt.title(f'Minimal Drift vs a for k_x = {kX} ')
plt.legend()
plt.tight_layout()
plt.show()

# Plot full drifts vs a
plt.figure(figsize=(6, 4))
plt.plot(a_grid, Dfull1, 'o-', label='D_full at Fold 1')
plt.plot(a_grid, Dfull2, 'x-', label='D_full at Fold 2')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('a')
plt.ylabel('D_full')
plt.title(f'Full Drift vs a for k_x={kX}')
plt.legend()
plt.tight_layout()
plt.show()

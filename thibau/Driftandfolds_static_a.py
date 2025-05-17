# -----------------------------------------------------------
# Extended GSPT diagnostic script
#  • Uses Hill-type f(x,a) and g(x) as before
#  • Adds slow-variable dynamics ȧ = (H(y) - a)/δ
#  • Computes the *full* drift   D = Fy*ŷ + Fa*ȧ  at each fold
# -----------------------------------------------------------
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ---------- parameters (edit to match your system) ----------
# fast–slow switch parameters (same as previous example)
p = dict(b=1.0, K=1.0, n=5, ap=0.1, bp=1.0, Kp=1.0, m=5)
a      = 0.3      # current value of a (treated as parameter when locating folds)
kX     = 1.7      # production rate in ẏ
delta  = 1.0      # timescale of ȧ (ε ≪ δ ≪ 1)

# parameters for H(y) from Eq. 8
abar      = 0.2
Delta_a   = 0.3
k_switch  = 4.0
Xc        = 1.0
# ------------------------------------------------------------

# --- Hill functions ---
def f(x, a=a):
    return a + p['b'] * x**p['n'] / (p['K']**p['n'] + x**p['n'])

def g(x):
    return p['ap'] + p['bp'] * p['Kp']**p['m'] / (p['Kp']**p['m'] + x**p['m'])

# --- Helper functions ---
def df_dx(x, a=a, h=1e-8):
    return (f(x+h, a) - f(x-h, a)) / (2*h)

def dg_dx(x, h=1e-8):
    return (g(x+h) - g(x-h)) / (2*h)

def H(y):
    return abar + Delta_a * np.tanh(k_switch * (y - Xc))

# fast subsystem
def F_eq(vars):
    x, y = vars
    return (y - x)*f(x) - x*g(x)

def dFdx_eq(vars):
    x, y = vars
    return -f(x) + (y - x)*df_dx(x) - g(x) - x*dg_dx(x)

def fold_system(vars):
    return [F_eq(vars), dFdx_eq(vars)]

# locate folds
initial_guesses = [(0.2, 0.5), (1.5, 1.2)]
folds = []
for guess in initial_guesses:
    sol, info, ier, msg = fsolve(fold_system, guess, full_output=True)
    if ier == 1:
        folds.append(sol)
    else:
        print("Warning: fsolve did not converge for guess", guess, ":", msg)
folds = np.array(folds)

# derivatives of F wrt y and a (analytic)
def Fy(x, a):
    return f(x, a)                # ∂F/∂y = f(x,a)

def Fa(x, y):
    # ∂F/∂a = (y - x)*∂f/∂a,   and ∂f/∂a = 1
    return (y - x)

# evaluate drift
print(f"Full drift analysis (a = {a}, δ = {delta})\n")
signs_full, signs_min = [], []
for i, (x_f, y_f) in enumerate(folds, 1):
    # minimal drift
    D_min = kX - x_f*y_f
    # full drift
    Dy = kX - x_f*y_f
    Da = (H(y_f) - a)/delta
    D_full = Fy(x_f, a)*Dy + Fa(x_f, y_f)*Da
    
    signs_full.append(np.sign(D_full))
    signs_min .append(np.sign(D_min))
    
    print(f"Fold {i}:  x = {x_f:.4f}, y = {y_f:.4f}")
    print(f"   D_min  = {D_min:+.4e}  ({'+' if D_min>0 else '-' if D_min<0 else '0'})")
    print(f"   D_full = {D_full:+.4e}  ({'+' if D_full>0 else '-' if D_full<0 else '0'})\n")

# decide based on full drift
if signs_full[0]*signs_full[1] < 0:
    verdict = "Opposite signs → limit cycle predicted (full criterion)."
else:
    verdict = "Same signs → no relaxation limit cycle (full criterion)."
print(verdict)

# -----------------------------------------------------------------
# Plot: critical manifold and folds
x_vals = np.linspace(0, max(folds[:,0])*1.2, 400)
y_null = x_vals + (x_vals * g(x_vals) / f(x_vals))  # from F=0

plt.figure(figsize=(6,5))
plt.plot(x_vals, y_null, 'b', label='Critical manifold F=0')
plt.scatter(folds[:,0], folds[:,1], c='r', s=60, marker='X', label='Folds')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Critical manifold & folds  (a={a:.2f})')
plt.legend()
plt.tight_layout()
plt.show()


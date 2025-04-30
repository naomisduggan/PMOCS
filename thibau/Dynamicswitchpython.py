# Dynamic Bistable Oscillatory System in Python
# Manual animation with ultra-precise solver settings

# Import dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root_scalar
except ImportError as e:
    print("Error importing dependencies:", e)
    print("Install with: pip install numpy matplotlib scipy")
    raise

# Enable interactive plotting
plt.ion()

# PARAMETERS & HILL FUNCTIONS
def g(x, p):
    return p.ap + p.bp * p.Kp**p.m / (p.Kp**p.m + x**p.m)

def f(x, a, p):
    return a + p.b * x**p.n / (p.K**p.n + x**p.n)

class Params:
    b = 1; K = 1; n = 5
    ap = 0.1; bp = 1; Kp = 1; m = 5
p = Params()

# Constants
a0, epsilon, kX = 0.3, 0.05, 1.6

da, delta, kappa = 0.0, 100, 5

# STATIC RESPONSE CURVE UTILITY
def ResponseOneEnzymeXt(a, p):
    f_loc = lambda x: f(x, a, p)
    g_loc = lambda x: g(x, p)
    xt_try = np.linspace(0, 5, 500)
    xv = np.full_like(xt_try, np.nan)
    for i, xt in enumerate(xt_try):
        fun = lambda x: f_loc(x)*(xt - x) - g_loc(x)*x
        try:
            sol = root_scalar(fun, bracket=[1e-8, max(1e-8, xt-1e-8)], method='bisect')
            xv[i] = sol.root
        except ValueError:
            continue
    valid = ~np.isnan(xv)
    xtv = xt_try[valid]; xv_valid = xv[valid]
    dFdx = np.gradient(f_loc(xv_valid)*(xtv - xv_valid) - g_loc(xv_valid)*xv_valid, xv_valid)
    idx = np.where(np.diff(np.sign(dFdx)))[0]
    if len(idx) >= 2:
        folds = np.vstack(([xtv[idx[0]], xv_valid[idx[0]]], [xtv[idx[1]], xv_valid[idx[1]]]))
    else:
        folds = np.full((2,2), np.nan)
    return xtv, xv_valid, folds

# Compute static nullcline folds
xtv_ref, xv_ref, folds = ResponseOneEnzymeXt(a0, p)
Xc = np.nanmean(folds[:,0])

# ODE system
def ddefun(t, y):
    X, XT, A = y
    dX = (f(X, A, p)*(XT - X) - g(X, p)*X) / epsilon
    dXT = kX - X*XT
    dA = (a0 + da*np.tanh(kappa*(XT - Xc)) - A) / delta
    return [dX, dXT, dA]

# Integrate two trajectories with ultra-tight tolerances
t_span = (0, 500)
# Dense sampling for ultra-smooth curves
t_eval = np.linspace(0, 500, 20000)
solver_kwargs = dict(method='DOP853', t_eval=t_eval, rtol=1e-12, atol=1e-15)
sol1 = solve_ivp(ddefun, t_span, [0, 0, a0], **solver_kwargs)
sol2 = solve_ivp(ddefun, t_span, [1, 1.8, a0], **solver_kwargs)

t = sol1.t
X1, XT1, A = sol1.y
X2, XT2 = sol2.y[0], sol2.y[1]

# Phase-plane setup
fig, ax = plt.subplots(figsize=(8,6))
grid_n = 6
gridX = np.linspace(0, 5, grid_n); gridXT = np.linspace(0, 5, grid_n)
Xg, XTg = np.meshgrid(gridX, gridXT)
V = kX - Xg*XTg
x_plot = np.linspace(1e-4, 5, 800)
frames = np.linspace(0, len(t)-1, 500, dtype=int)

for k in frames:
    ax.clear()
    a_t = A[k]
    U = (f(Xg, a_t, p)*(XTg - Xg) - g(Xg, p)*Xg) / epsilon
    # Sparse, short arrows
    ax.quiver(XTg, Xg, V, U, angles='xy', scale=1000, width=0.002, color='0.6')
    # Dynamic nullcline (red)
    xt_nc = x_plot + (g(x_plot, p)*x_plot) / f(x_plot, a_t, p)
    ax.plot(xt_nc, x_plot, 'r-', lw=2)
    # Static nullcline (blue dashed)
    ax.plot(kX/x_plot, x_plot, 'b--', lw=2)
    # Smooth trajectories
    ax.plot(XT1[:k], X1[:k], 'k-', lw=1.5)
    ax.plot(XT2[:k], X2[:k], 'g--', lw=1.5)
    ax.set(xlim=(0,5), ylim=(0,5), xlabel='X_T', ylabel='X',
           title=f't = {t[k]:.1f}, a = {a_t:.2f}')
    ax.grid(True)
    plt.pause(0.015)

# Hold final frame
plt.ioff()
plt.show()

# Static analysis plots
plt.figure(); plt.semilogx(t, A, lw=1.5)
plt.xlabel('Time'); plt.ylabel('a(t)'); plt.title('Switch Parameter'); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure();
for lab, cur, style in zip(['X1','XT1','X2','XT2'], [X1,XT1,X2,XT2], ['b-','r-','g--','m--']):
    plt.plot(t, cur, style, lw=1.5, label=lab)
plt.xlabel('Time'); plt.ylabel('Concentration'); plt.legend()
plt.title('Time Series'); plt.grid(True)
plt.tight_layout(); plt.show()

# Basic tests
if __name__ == '__main__':
    xtv2, xv2, f2 = ResponseOneEnzymeXt(a0, p)
    assert xtv2.shape == xv2.shape, 'Lengths mismatch'
    assert f2.shape == (2,2), 'Fold shape incorrect'
    print('All tests passed.')

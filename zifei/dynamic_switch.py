import numpy as np
import matplotlib.pyplot as plt
import changingswitches as cs

# Hill functions
def f(x, a, p):
    return a + p['b'] * x**p['n'] / (p['K']**p['n'] + x**p['n'])
def g(x, p):
    return p['ap'] + p['bp'] * p['Kp']**p['m'] / (p['Kp']**p['m'] + x**p['m'])

def activation_function(a_mean, delta_a, k, X_T, X_c):
    return a_mean + delta_a * np.tanh(k * (X_T - X_c))

# parameters
p = dict(ap=0.1, b=1, bp=1, K=1, Kp=1, n=5, m=5)
a_mean = 0.3
delta_a = 0.2
kX = 0.2

# simulation parameters
dt = 0.01
XT_target = 4.0
steps = int(XT_target / (kX * dt)) + 1

# initial values
X0 = 0.0
XT0 = 0.0

# calculate Xc using static switch
switch = cs.ResponseOneEnzymeXt(a=a_mean, **p)
switch.setcontpars(0.01, 1000)
switch.setstart(0, 0, [1, 0])
switch.compute_responsecurve()
folds = switch.folds
if len(folds) >= 2:
    X_c = 0.5 * (folds[0][0] + folds[1][0])

plt.figure(figsize=(7,5))

for kappa, style, label in [
    (0, '-', r'$\kappa=0 (static switch)$'),
    (0.5, '--', r'$\kappa=0.5$'),
    (5, '-.', r'$\kappa=5$')
]:
    X = X0
    XT = XT0
    X_arr = []
    XT_arr = []
    for i in range(steps):
        a = activation_function(a_mean, delta_a, kappa, XT, X_c)
        dXdt = f(X, a, p) * (XT - X) - g(X, p) * X
        dXTdt = kX
        X += dXdt * dt
        XT += dXTdt * dt
        X_arr.append(X)
        XT_arr.append(XT)
        if XT >= XT_target:
            break
    plt.plot(XT_arr, X_arr, style, label=label)

# Step function (kappa -> infinity)
X = X0
XT = XT0
X_arr = []
XT_arr = []
for i in range(steps):
    a = a_mean + delta_a * (1 if XT >= X_c else -1)
    dXdt = f(X, a, p) * (XT - X) - g(X, p) * X
    dXTdt = kX
    X += dXdt * dt
    XT += dXTdt * dt
    X_arr.append(X)
    XT_arr.append(XT)
    if XT >= XT_target:
        break
plt.plot(XT_arr, X_arr, ':', color='black', label='step function ($\kappa\\to\infty$)')

plt.xlabel('$X_T$')
plt.ylabel('$X$')
# plt.title('Dynamic Switch Response Curves for Different $\kappa$')
plt.xlim(0, 4)
plt.legend()
plt.tight_layout()
plt.savefig('dynamic_switch.png', dpi=300)
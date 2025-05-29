import numpy as np
import matplotlib.pyplot as plt

def activation_function(a_mean, delta_a, k, X_T, X_c):
    return a_mean + delta_a * np.tanh(k * (X_T - X_c))

# Parameters
a_mean = 0.5
delta_a = 1.0
X_T = np.linspace(-10, 10, 500)
X_c = 0.0

# k=5, k=0.5
a_values_k5 = activation_function(a_mean, delta_a, 5, X_T, X_c)
a_values_k05 = activation_function(a_mean, delta_a, 0.5, X_T, X_c)
# Step function for k→∞
a_values_step = a_mean + delta_a * np.where(X_T > X_c, 1, -1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X_T, a_values_k5, label="$k=5$")
ax.plot(X_T, a_values_k05, label="$k=0.5$")
ax.plot(X_T, a_values_step, label="step function ($k\\to\\infty$)", linestyle='--', color='black')
ax.axhline(y=a_mean, color='grey', linestyle='--')
ax.axvline(x=X_c, color='grey', linestyle='--')

# a bar
ax.text(X_T[0], a_mean + 0.03, r"$\overline{a}$", va='bottom', ha='left', fontsize=14, color='grey')

ax.set_xticks([X_c])
ax.set_xticklabels([r"$X_c$"])
ax.set_yticks([])

ax.set_xlabel("$X_T$")
ax.set_ylabel("$a$")
ax.legend()
ax.grid(False)
ax.tick_params(axis='both', which='both', length=0)
plt.savefig('a(Xt).png')
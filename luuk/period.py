import numpy as np
import matplotlib.pyplot as plt
from dynamics import Dynamics

kwargs = {
    'a': 0.5,
    'da': 0.3,
    'kX': 1.6,
    'kappa': 5.0,
    'b': 1.0,
    'K': 1.0,
    'n': 5,
    'ap': 0.1,
    'bp': 1.0,
    'Kp': 1.0,
    'm': 5
}

fig, ax1 = plt.subplots()

eps = 0.5
deltabin = np.arange(0.01, 10, 0.10)

peds = []
for dlta in deltabin:
    dyn = Dynamics(delta=dlta, epsilon=eps, **kwargs)
    dyn.delta = dlta
    dyn.calc_traj()
    peds.append(dyn.period)

# plot
ax1.set_xlabel('$\delta$')

ax1.plot(deltabin, peds, zorder=2)
ax1.set_ylabel('Period')

ax2 = ax1.twinx()
ax2.plot(deltabin, abs(deltabin - eps), color='gray', linestyle='--', zorder=1)
ax2.set_ylabel('$|\delta - \epsilon|$')
plt.show()

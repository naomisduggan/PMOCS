## Demo file for the code that was used in the paper
## Dynamic bistable switches enhance robustness and accuracy of cell cycle transitions
## by Jan Rombouts and Lendert Gelens

import numpy as np
import matplotlib.pyplot as plt

# import package to draw bistable switches and utility functions
import changingswitches as cs
import csutils as csu


### compute bistable switch and draw for different values of a (cfr Fig.2E)
# draw response curve for fixed a
av = [0.15,0.3,0.6,1]

fig, ax = plt.subplots()
for i, a in enumerate(av):
    switch = cs.ResponseOneEnzymeXt(b=1, K=1., n=5, ap=0.1, bp=1, Kp=1., m=5, a=a)

    #continuation parameters
    switch.setcontpars(0.01,2000)
    # starting values of point and vector
    switch.setstart(0,0,[1,0])

    switch.compute_responsecurve()
    # after running compute_responsecurve(),
    # the switch object contains XT and X values and fold coordinates if there are
    l, = ax.plot(switch.xtv, switch.xv, label='$a = {}$'.format(a))
    if switch.folds:
        for f in switch.folds:
            ax.plot(f[0], f[1], 'ko')

ax.set_xlim(0,3.5)
ax.set_ylim(0,3)
ax.set_xlabel('$X_T$')
ax.legend()
ax.set_ylabel('X')
# plt.show()
plt.savefig('response_curve_1.png', dpi=300)

# draw response curve for fixed Xt
xt_list = [1.2, 1.6, 2.5]
fig, ax = plt.subplots()

for i, xt in enumerate(xt_list):
    switch = cs.ResponseOneEnzyme(
        b=1, K=1., n=5, ap=0.1, bp=1, Kp=1., m=5, Xt=xt
    )
    switch.setcontpars(0.01, 2000)
    switch.setstart(0, 0, [1, 0])  # a初值, x初值, 切向量
    switch.compute_responsecurve()
    ax.plot(switch.av, switch.xv, label=f'$X_T={xt}$')
    if switch.folds:
        for f in switch.folds:
            ax.plot(f[0], f[1], 'ko')

ax.set_xlabel('$a$')
ax.set_ylabel('$X$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 2.5)
ax.legend()
plt.tight_layout()
plt.savefig('response_curve_2.png', dpi=300)
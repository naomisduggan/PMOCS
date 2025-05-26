# type: ignore

# Imports
import numpy as np
import sympy as sp
import scipy as sc
# import pandas as pd
import matplotlib.pyplot as plt
import pickle

# from jitcode import y as yo, t as to, jitcode
from jitcsde import y as ys, jitcsde, UnsuccessfulIntegration

import csutils as csu
import changingswitches as cs

cflags = ['-std=c11', '-O3', '-ffast-math', '-g0', '-march=native', '-mtune=native', '-Wno-unknown-pragmas']

# switch parameters
p = {
    'b': 1.0,
    'K': 1.0,
    'n': 5,
    'ap': 0.1,
    'bp': 1.0,
    'Kp': 1.0,
    'm': 5
}

# arguments
a = 0.3
da = 0.3
kX = 1.6
abr = 5.0
epsilon = 0.5
delta = 100
tf = 150
y0 = [0,0,0]
sigmas = [0, 0.2, 0.3, 0.5]

# hill functions
def f(x, a):
    return a + p['b'] * x**p['n'] / ( p['K']**p['n'] + x**p['n'] )

def g(x):
    return p['ap'] + p['b'] * p['Kp']**p['m'] / ( p['Kp']**p['m'] + x**p['m'] )

# switch
def F(x, xc):
    return sp.tanh(abr * (x - xc))

# response to changing Xt, with fixed a
switch = cs.ResponseOneEnzymeXt(a=a, **p)
switch.setcontpars(h=0.01, totit=2000)
switch.setstart(xt=0, x=0, v=[0.1,0.1])
switch.compute_responsecurve()

Xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])

# sdxdt2 = [
#     ( f(ys(0), a + da * F(ys(1), Xc) ) * ( ys(1) - ys(0) ) - g( ys(0) ) * ys(0) ) / epsilon,
#     kX - ys(1) * ys(0)
#     ]

sdxdt =[
    ( f(ys(0), ys(2) ) * ( ys(1) - ys(0) ) - g(ys(0)) * ys(0) ) / epsilon,
    kX - ys(1) * ys(0),
    ( a + da * F(ys(1), Xc) - ys(2) ) / delta
]

fig, axs = plt.subplots(2, len(sigmas), figsize=(14,8))

for i, sigma in enumerate(sigmas):
    data = {}
    noise = [sigma, 0, 0]
    sde = jitcsde(sdxdt, noise)
    sde.set_integration_parameters(atol=1e-8, first_step=0.001, max_step=0.01, min_step=1e-13)
    sde.compile_C(extra_compile_args = cflags)

    # integrate
    sde.set_initial_value(y0, 0.0)
    tv = np.arange(sde.t, sde.t + tf, 0.01)
    Xv, XTv, av = np.array([sde.integrate(t) for t in tv]).T

    #m1, find peaks in Xv
    crs, _ = sc.signal.find_peaks(Xv, width=100, height=1.0)
    cts = np.take(tv, crs)
    prs = np.diff(cts)
    pavg = np.average(prs)
    pstd = np.std(prs)
    data['m1'] = {'tv': tv, 'Xv': Xv, 'XTv': XTv, 'av': av, 'prs': prs, 'pavg': pavg, 'pstd': pstd}


    #m2, find threshold crossings
    ctt = csu.getcrossingtimes_twothresholds(tv, Xv, switch.folds[1][1], switch.folds[0][1])
    ctt2 = [] # new list with up and down jumps in turn
    j = 0

    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1] and ctt[j][1]=='u':
            ctt2.append(ctt[j][0])
            break
        j+=1

    # now take turns adding up and down jumps
    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1]:
            ctt2.append(ctt[j][0])
        j+=1

    ctt2 = np.array(ctt2)
    prs = ctt2[2::2]-ctt2[:-2:2]
    pavg = np.average(prs)
    pstd = np.std(prs)
    data['m2'] = {'tv': tv, 'Xv': Xv, 'XTv': XTv, 'av': av, 'prs': prs, 'pavg': pavg, 'pstd': pstd}


    for k, v in data.items():
        p = np.round(v['pavg'], 2)
        cv = np.round(v['pstd'], 2)
        axs[0,i].hist(v['prs'], bins=30, alpha=0.5, label=f'{k}, $\\bar{{T}}$ = {p}, $s^2$ = {cv}')
        axs[0,i].set_title(f'Variation of Periods $\\sigma_X$ = {sigma} \n $\\epsilon$ = {epsilon}, $\\delta$ = {delta}, $k_X$ = {kX}')
        axs[0,i].set_xlabel('period')
        axs[0,i].legend(loc='upper left')

    axs[1,i].plot(data['m1']['tv'], data['m1']['Xv'], label='$X$')
    axs[1,i].plot(data['m1']['tv'], data['m1']['XTv'], label='$X_T$')
    axs[1,i].plot(data['m1']['tv'], data['m1']['av'], label='$a$')
    # axs[1,i].set_xlim(0,100)
    axs[1,i].set_xlabel('time')
    axs[1,i].legend(loc='upper right')

axs[0,0].set_ylabel('frequency')
axs[1,0].set_ylabel('concentration')

plt.tight_layout()
plt.show()
# plt.savefig(f'var-period-{kX}-{epsilon}-{delta}.png')

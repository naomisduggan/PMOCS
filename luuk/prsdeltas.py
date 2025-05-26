# type: ignore

# Imports
import numpy as np
import sympy as sp
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime

# from jitcode import y as yo, t as to, jitcode
from jitcsde import y as ys, jitcsde, UnsuccessfulIntegration
import changingswitches as cs

# for clang
cflags = ['-std=c11', '-O3', '-ffast-math', '-g0', '-march=native', '-mtune=native', '-Wno-unknown-pragmas']
now = datetime.date.today()
date = now.strftime('%y%m%d')

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
kX = 1.4
abr = 5.0
epsilon = 0.5

d1 = np.arange(0., 1.01, 0.01)[1:]
d2 = np.arange(1., 10.1, 0.1)
d3 = np.arange(10., 101., 1.)
dbins = [d1, d2, d3]

tf = 200
y0 = [0.,0.,0.]
sigmas = [0., 0.1]

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

dats = []

# for sigma in sigmas:
#     noise = [sigma, 0, 0]
#
#     for j, dbin in enumerate(dbins):
#         for delta in dbin:
#             sdxdt =[
#                 ( f(ys(0), ys(2) ) * ( ys(1) - ys(0) ) - g(ys(0)) * ys(0) ) / epsilon,
#                 kX - ys(1) * ys(0),
#                 ( a + da * F(ys(1), Xc) - ys(2) ) / delta
#             ]
#
#             sde = jitcsde(sdxdt, noise)
#             sde.set_integration_parameters(atol=1e-8, first_step=0.001, max_step=0.01, min_step=1e-13)
#             sde.compile_C(extra_compile_args = cflags)
#
#             # integrate
#             sde.set_initial_value(y0, 0.0)
#             tv = np.arange(sde.t, sde.t + tf, 0.01)
#             try:
#                 Xv, XTv, av = np.transpose([sde.integrate(t) for t in tv])
#             except UnsuccessfulIntegration as e:
#                 print(f'Error for delta = {delta} and sigma = {sigma}: {e}')
#                 continue
#
#             dat = {
#                 'switch': p,
#                 'a': a,
#                 'da': da,
#                 'kx': kX,
#                 'kappa': abr,
#                 'epsilon': epsilon,
#                 'group': j,
#                 'sigma': sigma,
#                 'delta': delta,
#                 'tv': tv,
#                 'Xv': Xv,
#                 'XTv': XTv,
#                 'av': av,
#             }
#             dats.append(dat)

# save df
df = pd.DataFrame(dats)
with open(f'prsdeltas-{date}-{kX}-{epsilon}.pkl', 'wb') as fl:
    pickle.dump(df, fl)

# load df
with open('prsdeltas-250526-1.45-0.5.pkl', 'rb') as fl:
    df = pickle.load(fl)

def get_prd(x, tv):
    pki, _ = sc.signal.find_peaks(x, width=100., height=1.0)
    cts = np.take(tv, pki)
    cxs = np.take(x, pki)
    crs = zip(cts, cxs)
    fprs = np.diff(cts[5:])
    fpravg = np.average(fprs)
    fprstd = np.std(fprs)
    iprs = np.diff(cts[:5])
    ipravg = np.average(iprs)
    iprstd = np.std(iprs)
    frqs = np.fft.fft(x)
    amps = np.abs(frqs)
    return crs, fprs, fpravg, fprstd, iprs, ipravg, iprstd, amps


dfs_prd = lambda row: pd.Series(get_prd(row['Xv'], row['tv']))
df[['crs', 'fprs', 'fpravg', 'fprstd', 'iprs', 'ipravg', 'iprstd', 'amps']] = df.apply(dfs_prd, axis=1)

# plot
fig, axs = plt.subplots(3, len(dbins), figsize=(7,9))
cls = ['red', 'green', 'blue', 'yellow', 'purple']


pdeltas = [0.1, 5, 50]
for pdelta, (i, dfa) in zip(pdeltas, df.groupby('group')):
    for j, (sig, dfb) in enumerate(dfa.groupby('sigma')):
        deltas = dfb['delta']
        fpravg = dfb['fpravg']
        fprstd = dfb['fprstd']
        ipravg = dfb['ipravg']

        axs[0,i].fill_between(deltas, fpravg - fprstd, fpravg + fprstd, color = cls[j], alpha=0.2, label=f'$s^2(T): \\sigma_X = {sig}$')
        axs[0,i].scatter(deltas, fpravg, s=2, color=cls[j], alpha=0.6, label=f'$\\overline{{T}}: \\sigma_X = {sig}$')
        axs[0,1].set_xlim(min(np.floor(deltas)), max(np.ceil(deltas)))
        axs[0,i].set_ylim(0,10)


        dfc = dfb.loc[dfb['delta'].sub(pdelta).abs().idxmin()]
        delta = dfc['delta']
        tv = dfc['tv']
        xv = dfc['Xv']
        xtv = dfc['XTv']
        av = dfc['av']

        alpha = 1 - 0.8*j
        lx = '$X$' if j == 0 else None
        lxt = '$X_T$' if j == 0 else None
        la = '$a$' if j == 0 else None

        axs[1,i].plot(tv, xv, color=cls[0], label=lx, alpha=alpha)
        axs[1,i].plot(tv, xtv, color=cls[1], label=lxt, alpha=alpha)
        axs[1,i].plot(tv, av, color=cls[2], label=la, alpha=alpha)
        axs[1,i].set_xlim(0,50)
        axs[1,i].set_ylim(0,5)
        axs[1,i].text(0.95, 0.95, f'$\\delta = {np.round(delta, 1)}$', transform=axs[1,i].transAxes, ha='right', va='top')

        axs[2,i].plot(xtv, xv, color=cls[j], alpha=alpha, label=f'$\\sigma_X = {sig}$')
        axs[2,i].text(0.95, 0.95, f'$\\delta = {np.round(delta, 1)}$', transform=axs[2,i].transAxes, ha='right', va='top')
        axs[2,i].set_ylim(0,2)
        axs[2,i].set_xlim(0,5)
        if i > 0:
            axs[0,i].set_yticklabels([])
            axs[1,i].set_yticklabels([])
            axs[2,i].set_yticklabels([])


axs[0,1].set_xlim(0,10)
axs[0,2].set_xlim()
axs[0,0].legend(loc='best')
axs[0,0].set_ylabel('time')
axs[0,1].set_xlabel('$\\delta$')

axs[1,0].legend(loc='upper left')
axs[1,0].set_ylabel('concentration')
axs[1,1].set_xlabel('$t$')

axs[2,0].legend(loc='upper left')
axs[2,0].set_ylabel('$X$')
axs[2,1].set_xlabel('$X_T$')

fig.suptitle(f'Relaxation (oscillations) of $X(t),\\, X_T(t),\\, a(t)$ for $k_X = {df['kx'][0]}, \\kappa = {df['kappa'][0]}, \\epsilon = {df['epsilon'][0]}$')

plt.tight_layout()
plt.show()
# plt.savefig(f'prsdeltas-{kX}-{epsilon}.png')

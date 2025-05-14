## Demo file for the code that was used in the paper
## Dynamic bistable switches enhance robustness and accuracy of cell cycle transitions
## by Jan Rombouts and Lendert Gelens

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.animation as animation

# import package to draw bistable switches and utility functions
import changingswitches as cs
import csutils as csu

# import package for simulation
from jitcsde import y as ys, t as ts, jitcsde

## Simulate
p = dict(b=1., K=1, n=5, ap=0.1, bp=1., Kp=1., m=5)

a = 0.3
da = 0.3

# Hill functions
def f(x,a):
    return a +p['b']*x**p['n']/(p['K']**p['n'] + x**p['n'])
def g(x):
    return p['ap']+p['bp']*p['Kp']**p['m']/(p['Kp']**p['m'] + x**p['m'])
def fx(x,a):
    return (x**(p['n']- 1)*p['b']*p['n'])/(p['K']**p['n'] + x**p['n']) - (x**(2*p['n'] - 1)*p['b']*p['n'])/(p['K']**p['n'] + x**p['n'])**2
def gx(x):
    return -(p['Kp']**p['m']*x**(p['m'] - 1)*p['bp']*p['m'])/(p['Kp']**p['m'] + x**p['m'])**2

def JacobiEigvals(xt,x,a):
    dXdtX = 1/epsilon*(fx(x, a)*(xt-x) - f(x,a) - gx(x)*x - g(x))
    dXdtXT = 1/epsilon*f(x,a)
    dXTdtX = -xt
    dXTdtXT = -x
    Jacobi = np.array([[dXdtX, dXdtXT],[dXTdtX, dXTdtXT]])
    eigvals, eigvecs = np.linalg.eig(Jacobi)
    return eigvals
    
# compute switch
switch= cs.ResponseOneEnzymeXt(a=a,**p)
switchdat = []

switch.setcontpars(0.01,2000)
switch.setstart(0,0,[0.1,0.1])
switch.compute_responsecurve()


kX = 1.6
kappa = 5.0
epsilon = 0.05
sigma = 0.0 # this is the noise strength


# threshold value used in a(X_T)
Xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])
print("Xc", Xc)
def F(x, xc):
    #function which returns values between -1 and 1
    return sp.tanh(kappa*(x-xc))
def Fn(x, xc):
    return np.tanh(kappa*(x-xc))

fig, axes = plt.subplots(1,2)


print("Start Simulation")
for i,delta in enumerate([100]):
    print("Start simulation delta=", delta)
    dxdt = [
        1/epsilon*(f(ys(0), ys(2))*(ys(1)-ys(0))- g(ys(0))*ys(0)), 
        kX-ys(1)*ys(0),
        1/delta*(a+da*F(ys(1), Xc)- ys(2))
            ]    

    ggg = [sigma, 0, 0] #noise
    ini = [0.1,0.1,0]
    sdesys = jitcsde(dxdt, ggg)
    sdesys.set_initial_value(ini,0.0)
    sdesys.set_integration_parameters(atol=1e-8,first_step=0.001, max_step=0.01,min_step=1e-13)
    timeseries = []
    tv = np.arange(sdesys.t, sdesys.t+1000, 0.01)
    for time in tv:
        timeseries.append(sdesys.integrate(time) )

    timeseries = np.array(timeseries)
    Xv = timeseries[:,0]
    XTv = timeseries[:,1]
    av = timeseries[:,2]


# Calculate Jacobi Matrix at fixed point


line_response, = axes[0].plot([], [], color='b')

# Plot phase planes, time series etc.
axes[0].set_xlabel('$X_T$')
axes[0].set_ylabel('$X$')
axes[0].set_xlim(0,9)
axes[0].set_ylim(0,2)
line_traj, = axes[0].plot([], [])
scat_fix = axes[0].scatter([], [])
lineX, = axes[1].plot([], [], label='$X$')
lineXT, = axes[1].plot([], [], label='$X_T$')
linea, = axes[1].plot([], [], label='$a$')
axes[0].plot(switch.xtv, [kX/xtv for xtv in switch.xtv])
text = axes[0].text(0.05, 0.95, '', transform=axes[0] .transAxes)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Concentration')
axes[1].legend()
axes[1].set_xlim(0,300)
axes[1].set_ylim(0,9)
axes[0].set_title(r'$\delta=$' + str(delta))


def update(frame):
    lineX.set_xdata(tv[:frame*50])
    lineX.set_ydata(Xv[:frame*50])
    lineXT.set_xdata(tv[:frame*50])
    lineXT.set_ydata(XTv[:frame*50])
    linea.set_xdata(tv[:frame*50])
    linea.set_ydata(av[:frame*50])
    
    switch.a = av[frame*50]
    switch.compute_responsecurve()
    line_response.set_xdata(switch.xtv)
    line_response.set_ydata(switch.xv)
    
    line_traj.set_xdata(XTv[:frame*50])
    line_traj.set_ydata(Xv[:frame*50])
    
    diff = abs(switch.xv-kX/switch.xtv)
    diff_min = min(diff)
    imin = np.where(diff == diff_min)[0]
    xtmin = switch.xtv[imin][0]
    xmin = switch.xv[imin][0]
    scat_fix.set_offsets([[xtmin, xmin]])
    text.set_text(JacobiEigvals(xtmin, xmin, switch.a)[0])
    
    
    return (lineX, lineXT, linea, line_traj, scat_fix, text)

ani = animation.FuncAnimation(fig=fig, func=update, interval=0.01)

plt.show()





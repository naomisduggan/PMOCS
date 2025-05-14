import numpy as np
import sympy as sp

import csutils as csu
import changingswitches as cs

from changingswitches import ResponseOneEnzymeXt
from jitcode import y, t, jitcode
from jitcxde_common import DEFAULT_COMPILE_ARGS
from abc import abstractmethod, abstractproperty

# in case of clang
cflags = DEFAULT_COMPILE_ARGS.copy() + ['-O3', '-ffast-math']
cflags.remove('-Ofast')

class ADyn():

    def __init__(self, **p):
        self.kappa = p['kappa']
        
    @abstractproperty
    def _xc(self):
        pass
        
    def _Fn(self, x):
        return np.tanh(self.kappa*(x-self._xc))

    def _F(self, x):
        return sp.tanh(self.kappa*(x-self._xc))


class XDyn(ResponseOneEnzymeXt):
    
    def __init__(self, **p):
        super().__init__(**p)

    def _f(self, x, a):
        return a + self.b * x ** self.n / (self.K ** self.n + x ** self.n)

    def _g(self, x):
        return self.ap + self.bp * self.Kp ** self.m / (self.Kp ** self.m + x ** self.m)


class Dynamics(XDyn, ADyn):

    def __init__(self, delta, epsilon, y0=[0,0,0], tf=20, **p):
        XDyn.__init__(self, **p)
        ADyn.__init__(self, **p)
        # jitcode.__init(self)
        self.setcontpars(0.01,2000)
        self.setstart(0,0,[0.1,0.1])
        self.compute_responsecurve()
        
        self.da = p['da']
        self.kX = p['kX']
        
        self.delta = delta
        self.epsilon = epsilon
        self.tf = tf
        self.y0 = y0
        self.sys = self._dynswitch
        
    @property
    def sys(self):
        return self._sys

    @sys.setter
    def sys(self, sys):
        self._sys = jitcode(sys)
        try:
            self._sys.compile_C(extra_compile_args = cflags)
        except:
            self._sys.compile_C(extra_compile_args = DEFAULT_COMPILE_ARGS)

    @property
    def _xc(self):
        return 0.5 * (self.folds[0][0] + self.folds[1][0])

    @property
    def _dynswitch(self):
        return [
            (self._f(y(0), y(2)) * (y(1) - y(0)) - self._g(y(0)) * y(0)) / self.epsilon,
            self.kX - y(1) * y(0),
            (self.a + self.da * self._F(y(1)) - y(2)) / self.delta
        ]
    
    def calc_traj(self):
        self._sys.set_initial_value(self.y0)
        self._sys.set_integrator('RK45')
        self.tv = np.arange(self._sys.t, self._sys.t + self.tf, 0.01)
        self.Xv, self.XTv, self.av = np.array([self._sys.integrate(t) for t in self.tv]).T

    @property
    def period(self):
        ct = csu.getcrossingtimes_twothresholds(self.tv, self.Xv, self.folds[1][1], self.folds[0][1])
        if len(ct)>3:
            return abs(ct[-1][0]-ct[-3][0])
        else:
            return 0

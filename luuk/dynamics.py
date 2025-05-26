import numpy as np
import sympy as sp

import csutils as csu
import changingswitches as cs

from changingswitches import ResponseOneEnzymeXt
from jitcode import y as yo, t as to, jitcode
from jitcsde import y as ys, t as ts, jitcsde

from jitcxde_common import DEFAULT_COMPILE_ARGS
from abc import ABC, abstractmethod, abstractproperty

# in case of clang
try:
    cflags = DEFAULT_COMPILE_ARGS.copy() + ['-O3', '-ffast-math']
    cflags.remove('-Ofast')
except ValueError as v:
    # raise Warning('Warning: {}'.format(v))
    pass

class ADyn():
        
    def _xc(self):
        return 0.5 * (self.folds[0][0] + self.folds[1][0])
        
    def _Fn(self, x):
        return np.tanh(self.kappa*(x-self._xc))

    def _F(self, x):
        return sp.tanh(self.kappa*(x-self._xc))


class Dyn():
    
    def __init__(self, **p):
        self.kappa = p.get('kappa')

        ResponseOneEnzymeXt.__init__(**p)
        self.setcontpars(0.01, 2000)
        self.setstart(0, 0, [0.1, 0.1])
        self.compute_responsecurve()


    @property
    def _f(self, x, a):
        return a + self.b * x ** self.n / (self.K ** self.n + x ** self.n)

    @property
    def _g(self, x):
        return self.ap + self.bp * self.Kp ** self.m / (self.Kp ** self.m + x ** self.m)

    @property
    def _xc(self):
        return 0.5 * (self.folds[0][0] + self.folds[1][0])
    
    @property
    def _Fn(self, x):
        return np.tanh(self.kappa*(x-self._xc))

    @property
    def F(self, x):
        return sp.tanh(self.kappa*(x-self._xc))


class Dynamics(XDyn):

    def __init__(self, delta, epsilon, y0, tf, **p):
        ADyn().__init__(**p)
        XDyn().__init__(**p)

        self.da = p.get('da')
        self.kX = p.get('kX')

        self.delta = delta
        self.epsilon = epsilon

        self.tf = tf
        self.y0 = y0

        self.tv = None
        self.Xv = None
        self.XTv = None
        self.av = None

        self.sys = self._dynswitch
        self.tv = np.arange(self._sys.t, self._sys.t + self.tf, 0.01)
    
    @abstractproperty
    def _dynswitch(self):
        pass

    @abstractproperty
    def sys(self):
        pass

    def calc_traj(self):
        self.Xv, self.XTv, self.av = np.array([self.sys.integrate(t) for t in self.tv]).T

    def calc_period(self):
        try:
            ct = csu.getcrossingtimes_twothresholds(self.tv, self.Xv, self.folds[1][1], self.folds[0][1])
            if len(ct) > 3:
                return abs(ct[-1][0]-ct[-3][0])
            else:
                return 0
        except TypeError:
            raise ValueError('Calculate trajectory first')

class ODESys(Dynamics):

    def __init__(self, delta, epsilon, y0, tf, **p):
        Dynamics.__init__(delta, epsilon, y0, tf, **p)

    @property
    def sys(self):
        return self._sys

    @sys.setter
    def sys(self, sys):
        self._sys = jitcode(sys)
        self._sys.set_initial_value(self.y0)
        self._sys.set_integrator('RK45')
        try:
            self._sys.compile_C(extra_compile_args = cflags)
        except:
            self._sys.compile_C(extra_compile_args = DEFAULT_COMPILE_ARGS)

    @property
    def _dynswitch(self):
        return [
            (self._f(yo(0), yo(2)) * (yo(1) - yo(0)) - self._g(yo(0)) * yo(0)) / self.epsilon,
            self.kX - yo(1) * yo(0),
            (self.a + self.da * self._F(yo(1)) - yo(2)) / self.delta
        ]


class SDESys(Dynamics):

    def __init__(self, delta, epsilon, y0, tf, **p):
        super().__init__(delta, epsilon, y0, tf, **p)
        self.noise = p.get('noise')
        
    @property
    def sys(self):
        return self._sys

    @sys.setter
    def sys(self, sys):
        self._sys = jitcsde(sys, self.noise)
        self._sys.set_initial_value(self.y0)
        self._sys.set_integrator('RK45')
        try:
            self._sys.compile_C(extra_compile_args = cflags)
        except:
            self._sys.compile_C(extra_compile_args = DEFAULT_COMPILE_ARGS)
    
    @property
    def _dynswitch(self):
        return [
            (self._f(ys(0), ys(2)) * (ys(1) - ys(0)) - self._g(ys(0)) * ys(0)) / self.epsilon,
            self.kX - ys(1) * ys(0),
            (self.a + self.da * self._F(ys(1)) - ys(2)) / self.delta
        ]
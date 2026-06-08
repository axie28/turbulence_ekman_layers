# K-theory parametrization

import numpy as np


class Kmodel:
    K0 = 5.0  # m^2 /s, default turbulent viscosity
    lm0 = 5.0  # m, default mixing length
    z0 = 5.0  # m, roughness length for near-surface regularization
    # tau = 3600.0 * 4.0  # transient, if any

    def __init__(self, lm_outer=10.0 * lm0):
        self.lm_outer = lm_outer  # outer scale
        self.lm0 = 0.1 * self.lm_outer  # reference mixing length, m

    def reference(self, **kwargs):
        return self.K0, self.lm0

    def constant(self, z, **kwargs):
        return np.full_like(z, self.K0), self.lm0

    def stepwise(self, z, **kwargs):
        print('to be done')
        quit()

    def Prandtl(self, z, vel, **kwargs):
        print('to be done')
        quit()

    def Blackadar(self, z, vel, **kwargs):
        print('to be done')
        quit()

    def Tke(self, z, vel, tke, **kwargs):
        print('to be done')
        quit()
# K-theory parametrization

import numpy as np


class Kmodel:
    K0 = 5.0  # m^2 /s, default turbulent viscosity
    lm0 = 5.0 # m, default mixing length
    z0 = 5.0  # m, roughness length for near-surface regularization
    # b_layer = 300 # m, boundary layer
    # tau = 3600.0 * 4.0  # transient, if any

    def __init__(self, lm_outer=10.0 * lm0): 
        self.lm_outer = lm_outer  # outer scale
        self.lm0 = 0.1 * self.lm_outer  # reference mixing length, m

    def reference(self, **kwargs):
        return self.K0, self.lm0

    def constant(self, z, **kwargs):
        return np.full_like(z, self.K0), self.lm0

    def stepwise(self, z, **kwargs): # strong stable conditions
        K = np.empty_like(z)
        K.fill(self.K0)
        K[ z>self.lm_outer ] = 0
        return K, self.lm0


    def Prandtl(self, z, vel, **kwargs):
        u, v = vel
        dudz = np.gradient(u, z)
        dvdz = np.gradient(v, z)
        S = np.sqrt(dudz**2 + dvdz**2)  #shear calculation
        Kt = self.lm0**2 * S  #Prandtl estimation of K constant
        return Kt, self.lm0

    def Blackadar(self, z, vel, **kwargs):
        u, v = vel
        kappa = 0.41

        lm_outer = 0.1 *self.lm_outer  #setting outer boundary assuming constant of c = 0.1
        lm_inner = np.maximum(kappa * z, self.z0)  #inner boundary
        lm = lm_outer * lm_inner / (lm_inner + lm_outer)  #blend outer and inner regions

        dudz = np.gradient(u, z)
        dvdz = np.gradient(v, z)
        S = np.sqrt(dudz ** 2 + dvdz ** 2)  #shear calculation

        Kt = lm**2 * S

        return Kt, lm


    def Tke(self, z, vel, tke, **kwargs):
        #constants
        c = 0.55
        kappa = 0.41

        lm_outer = 0.1 * self.lm_outer  # setting outer boundary assuming constant of c = 0.1
        lm_inner = np.maximum(kappa * z, self.z0)  # inner boundary
        lm = lm_outer * lm_inner / (lm_inner + lm_outer)  # blend outer and inner regions

        Kt = c * lm * np.sqrt(np.maximum(tke, 1e-12))

        nref = np.abs(z[:] - self.z0 / kappa).argmin()
        Kt[:nref] = lm[:nref]**2 * np.sqrt(np.gradient(vel[0],z)[:nref]**2+np.gradient(vel[1],z)[:nref]**2) #near surface regularization

        return Kt, lm
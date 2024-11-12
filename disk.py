import numpy as np
import astropy.constants as c
from inputFile import (
    MS,mu,alphaTurb,vfrag
)

class Disk:

    def __init__(self, model):
        self.model = model
        self.st_mono = self.calc_st_mono()
        self.d2gRho = self.calc_d2gRho()
        self.d2gRho_vortex = self.calc_d2gRho_vortex()
        self.d2gRho_vortex_growth, self.rhoDust_vortex_growth, self.st_vortex_growth = self.calc_d2gRho_vortex_growth()

    def calc_st_mono(self):
        ind = np.argmax(self.model.rho_dust_poly, axis=2)
        return np.take_along_axis(self.model.st_poly, ind[..., np.newaxis], axis=2).squeeze(axis=2)
    
    def calc_d2gRho(self):
        return self.model.rho_dust_poly.sum(-1)/self.model.rho_gas
    
    def calc_d2gRho_vortex(self):
        d2gSigma = self.model.sigma_dust_poly.sum(-1)/self.model.sigma_gas
        d2g = d2gSigma * (self.st_mono/alphaTurb + 1.0)**1.5
        return d2g

    def calc_d2gRho_vortex_growth(self):

        def _epsilon(Z, St, alpha):
            x = Z**2 * (1 + St / alpha) / (1 + St)
            return (x + np.sqrt(x**2 + 4 * x *(1+St))) / 2

        kB = c.k_B.cgs.value
        mH = c.u.cgs.value
        G = c.G.cgs.value
    
        Cs = (kB * self.model.temp / (mu * mH))**(1/2)
        Omega = np.sqrt(G*MS/self.model.r**3)
        rho_R = 9*Omega**2 / (4*np.pi*G)
        d2gSigma = self.model.sigma_dust_poly.sum(-1)/self.model.sigma_gas

        # Initialize loop variables
        Z = Z0 = d2gSigma
        St = St0 = vfrag**2 / (3 * alphaTurb * Cs**2)
        eps = _epsilon(Z, St, alphaTurb)
        Omega_v = 0.5 # Kida solution for X = 4
        rhoDust = self.d2gRho * self.model.rho_gas

        update_mask = (St < 1) & (rhoDust < rho_R)
        while np.any(update_mask):
            Z_max = Z0 * (1 + St / alphaTurb)
            St_frag = St0 * (1 + St + eps) / (1 + St)

            # Timestep for this iteration
            t_Z = 1 / Omega_v
            t_St = 1 / (Z * Omega_v)
            dt = 0.1 * np.minimum(t_Z, t_St)

            # Z and St both grow together
            Z[update_mask] = np.minimum(Z_max[update_mask], Z[update_mask] * np.exp(dt[update_mask] / t_Z))
            St[update_mask] = np.minimum(St_frag[update_mask], St[update_mask] * np.exp(dt[update_mask] / t_St[update_mask]))
            eps[update_mask] = _epsilon(Z[update_mask], St[update_mask], alphaTurb)
            rhoDust[update_mask] = eps[update_mask] * self.model.rho_gas[update_mask]

            update_mask = (St < 1) & (rhoDust < rho_R)

        return eps, rhoDust, St
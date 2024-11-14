import numpy as np
import astropy.constants as c
from inputFile import (
    MS,mu,alphaTurb,vfrag
)

class Disk:
    """
    This class handles the calculation of dust-to-gas ratios
    and monodisperse Stokes numbers inside and outside a
    vortex
    """
    def __init__(self, model):
        self.model = model
        self.St_mono = self.calc_St_mono()
        self.d2gRho = self.calc_d2gRho()
        self.d2gRho_vortex = self.calc_d2gRho_vortex()
        self.d2gRho_vortex_growth, self.rhoDust_vortex_growth, self.St_vortex_growth = self.calc_d2gRho_vortex_growth()

    def calc_St_mono(self):
        """
        Calculates a representative Stokes number, taken to
        be the Stokes number at the peak of the dust density
        versus Stokes number distribution
        """
        ind = np.argmax(self.model.rho_dust_poly, axis=2)
        return np.take_along_axis(self.model.St_poly, ind[..., np.newaxis], axis=2).squeeze(axis=2)
    
    def calc_d2gRho(self):
        """
        Calculates the dust-to-gas ratio outside a vortex
        """
        return self.model.rho_dust_poly.sum(-1)/self.model.rho_gas
    
    def calc_d2gRho_vortex(self):
        """
        Calculates the maximum dust-to-gas midplane density
        ratio inside a Kida vortex, using eq.8 of Lyra et 
        al. (2024)
        """
        d2gSigma = self.model.sigma_dust_poly.sum(-1)/self.model.sigma_gas
        d2g = d2gSigma * (self.St_mono/alphaTurb + 1.0)**1.5
        return d2g

    def calc_d2gRho_vortex_growth(self):
        """
        Calculates the maximum Stokes number, dust midplane
        density and dust-to-gas midplane density ratio 
        inside a vortex when mass loading, dust growth
        and dust concentration is taken into account
        Description in Eriksson et al. (in prep.)
        """
        def _epsilon(Z, St, alpha):
            x = Z**2 * (1 + St / alpha) / (1 + St)
            return (x + np.sqrt(x**2 + 4 * x *(1+St))) / 2
        Cs = (c.k_B.cgs.value * self.model.temp / (mu * c.u.cgs.value))**(1/2)
        Omega = np.sqrt(c.G.cgs.value*MS/self.model.r**3)
        rho_R = 9*Omega**2 / (4*np.pi*c.G.cgs.value)
        d2gSigma = self.model.sigma_dust_poly.sum(-1)/self.model.sigma_gas
        Z = Z0 = d2gSigma
        St = St0 = vfrag**2 / (3 * alphaTurb * Cs**2)
        eps = _epsilon(Z, St, alphaTurb)
        Omega_v = 0.5 # Kida solution for X = 4
        rhoDust = self.d2gRho * self.model.rho_gas
        max_iterations = 1000
        iteration_count = 0
        update_mask = (St < 1) & (rhoDust < rho_R[None,:])
        while np.any(update_mask) and iteration_count < max_iterations :
            Z_max = Z0 * (1 + St / alphaTurb)
            St_frag = St0 * (1 + St + eps) / (1 + St)
            t_Z = 1 / Omega_v
            t_St = 1 / (Z * Omega_v)
            dt = 0.1 * np.minimum(t_Z, t_St)
            Z[update_mask] = np.minimum(Z_max[update_mask], Z[update_mask] * np.exp(dt[update_mask] / t_Z))
            St[update_mask] = np.minimum(St_frag[update_mask], St[update_mask] * np.exp(dt[update_mask] / t_St[update_mask]))
            eps[update_mask] = _epsilon(Z[update_mask], St[update_mask], alphaTurb)
            rhoDust[update_mask] = eps[update_mask] * self.model.rho_gas[update_mask]
            iteration_count += 1
            update_mask = (St < 1) & (rhoDust < rho_R[None,:])
        if iteration_count >= max_iterations:
            print("Reached maximum number of iterations in while loop")    
        return eps, rhoDust, St
        
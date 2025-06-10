import astropy.constants as c
import input_loader
import numpy as np

class Disk:
    """
    This class handles the calculation of dust-to-gas ratios
    and representative Stokes numbers inside and outside a
    vortex
    """
    def __init__(self, model):
        self.model = model
        self.path = self.model.path
        input_file = input_loader.load_input_file(self.path)
        self.Mstar = input_file.Mstar
        self.mu = input_file.mu
        self.alphaTurb = input_file.alphaTurb
        self.vfrag = input_file.vfrag
        self.lifeVortex = input_file.lifeVortex
        self.St_mono = self.calc_St_aver() # We use the density-weighted average St
        self.d2gRho = self.calc_d2gRho() # d2g in no vortex case
        self.d2gRho_vortex = self.calc_d2gRho_vortex() # d2g in case VC
        self.d2gRho_vortex_growth, self.rhoDust_vortex_growth, self.St_vortex_growth = self.calc_d2gRho_vortex_growth() # d2g and St in case VCG

    def calc_St_mono(self):
        """
        Calculates a representative Stokes number, taken to
        be the Stokes number at the peak of the dust density
        versus Stokes number distribution
        """
        ind = np.argmax(self.model.rho_dust_poly, axis=2)
        return np.take_along_axis(self.model.St_poly, ind[..., np.newaxis], axis=2).squeeze(axis=2)

    def calc_St_aver(self):
        """
        Calculates a representative Stokes number, taken to
        be the density-weighted average Stokes
        """
        return np.sum(self.model.St_poly * self.model.rho_dust_poly, axis=2) / np.sum(self.model.rho_dust_poly, axis=2)
    
    def calc_d2gRho(self):
        """
        Calculates the dust-to-gas ratio outside a vortex
        """
        return self.model.rho_dust_poly.sum(-1) / self.model.rho_gas
    
    def calc_d2gRho_vortex(self):
        """
        Calculates the maximum dust-to-gas midplane density
        ratio inside a Kida vortex, using Eq.8 of Lyra et
        al. (2024)
        Case VC
        """
        d2gSigma = self.model.sigma_dust_poly.sum(-1) / self.model.sigma_gas
        St = self.St_mono.copy()
        d2g = d2gSigma * (St / self.alphaTurb + 1.0)**1.5
        return d2g

     def calc_d2gRho_vortex_growth(self):
        """
        Calculates the maximum Stokes number, dust midplane
        density and dust-to-gas midplane density ratio
        inside a vortex when mass loading, dust growth
        and dust concentration are taken into account.
        Description in Eriksson et al. (2025), model from
        Carrera et al. (2025).
        """
        def _epsilon(Z, St, alpha):
            x = Z**2 * (1 + St / alpha) / (1 + St)
            return (x + np.sqrt(x**2 + 4 * x *(1+St))) / 2
        Cs = (c.k_B.cgs.value * self.model.temp / (self.mu * c.u.cgs.value))**(1/2)
        Omega = np.sqrt(c.G.cgs.value * self.Mstar / self.model.r**3)
        rho_R = 9 * Omega**2 / (4 * np.pi * c.G.cgs.value)
        d2gSigma = self.model.sigma_dust_poly.sum(-1) / self.model.sigma_gas
        Z0 = d2gSigma.copy()
        Z = Z0.copy()
        St = self.St_mono.copy()
        St_frag_old = self.vfrag**2 / (3 * self.alphaTurb * Cs**2)
        eps = self.d2gRho.copy()
        rhoDust = eps * self.model.rho_gas
        Omega_1 = 1.0
        Omega_v = np.full(Z.shape, 0.5 * Omega_1)

        tmax = self.lifeVortex/Omega_1 
        t = np.full(Z.shape, 0.0)
        update_mask = (St < 1) & (rhoDust < rho_R[None,:]) & (t < tmax)
        while np.any(update_mask):
            Z_max = Z0 * (1 + St / self.alphaTurb)
            St_frag = St_frag_old * (1 + St + eps) / (1 + St)
            t_Z = 1 / (St * Omega_v)
            t_St = 1 / (Z * Omega_v)
            dt = 0.1 * np.minimum(t_Z, t_St)
            Z[update_mask] = np.minimum(Z_max[update_mask], Z[update_mask] * np.exp(dt[update_mask] / t_Z[update_mask]))
            St[update_mask] = np.minimum(St_frag[update_mask], St[update_mask] * np.exp(dt[update_mask] / t_St[update_mask]))
            eps[update_mask] = _epsilon(Z[update_mask], St[update_mask], self.alphaTurb)
            rhoDust[update_mask] = eps[update_mask] * self.model.rho_gas[update_mask]
            t[update_mask] += dt[update_mask]
            update_mask = (St < 1) & (rhoDust < rho_R[None,:]) & (t < tmax)
        if (t.max() >= tmax):
            print("Reached t>tmax in while loop")
        return eps, rhoDust, St
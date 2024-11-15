import numpy as np
import astropy.constants as c
from inputFile import (
    MS,mu
)

class Planetesimal:
    """
    This clas handles the formation of planetesimals and
    the mass of the embryos that form
    """
    def __init__(self, model, d2gSt):
        self.model = model
        self.d2gSt = d2gSt
        self.rocheDensity()
        self.massVortex()
    
    def rocheDensity(self):
        Omega = np.sqrt(c.G.cgs.value*MS/self.model.r**3)
        self.rho_R = 9*Omega**2 / (4*np.pi*c.G.cgs.value)

    def SI_Lim2024a(self, St):
        """
        Calculates the critical dust-to-gas midplane 
        density ratio for the streaming instability
        to develop according to eq.16 of Lim et al.
        (2024a)
        """
        eps_crit = 10**( 0.42 * np.log10(St)**2 + 0.72 * np.log10(St) + 0.37 )
        return eps_crit

    def SI_Lim2024b(self, St):
        """
        Calculates the critical dust-to-gas midplane 
        density ratio for the streaming instability
        to develop according to eq.15 of Lim et al.
        (2024b)
        A description of how the criteria is converted
        from surface densities to midplane densities is
        found in Eriksson et al. (in prep.)
        """
        Z_crit = 10**( 0.1 * np.log10(St)**2 + 0.07 * np.log10(St) - 2.36 )
        xi = 100 * Z_crit / np.sqrt(1 + St)
        eps_crit = (xi**2 + np.sqrt(xi**4 + 4 * xi**2 * (1 + St))) / 2
        return eps_crit

    def planForm(self, eps, eps_crit):
        """
        Check if the chosen SI criteria or the Roche
        density is reached
        """
        pf = np.array((eps > eps_crit) | (eps * self.model.rho_gas > self.rho_R[None,:]), dtype=int)
        return pf
    
    def planForm_roche(self, eps):
        """
        Check if the Roche density is reached
        """
        pf = np.array((eps * self.model.rho_gas > self.rho_R[None,:]), dtype=int)
        return pf
    
    def massVortex(self):
        """
        Calculates the total dust mass trapped inside a
        vortex using eq.65 from Lyra et al. (2013)
        We assume a Kida vortex with chi=4
        """
        def _scaleFunction(chi):
            omega_v = 3/2 / (chi - 1)
            xi_plus = 1 + chi**(-2)
            f2chi = 2 * omega_v * chi - xi_plus**(-1) * (2 * omega_v**2 + 3)
            return np.sqrt(f2chi)
        chi = 4
        a = self.model.r
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        temp = self.model.temp
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega[None,:]
        rho_dust = self.model.rho_dust_poly.sum(-1)
        fchi = _scaleFunction(chi)
        Hg = H / fchi
        self.Mvortex = (2 * np.pi)**(3/2) * rho_dust * chi * H * Hg**2

    def embryoMass_vortex(self, pf):
        """
        Calculates the embryo mass using eq.X and assuming
        that the embryo mass is 10x the standard mass of
        a planetesimal forming via the SI
        Checks that embryo mass is not larger than the total
        mass of trapped dust in the vortex
        """
        gamma = 1
        Zfil = 0.1
        Omega = (c.G.cgs.value * MS / self.model.r**3)**(1/2)
        Cs = (c.k_B.cgs.value * self.model.temp / (mu * c.u.cgs.value))**(1/2)
        H = Cs / Omega[None,:]
        h = H / self.model.r[None,:]
        mp = 5e-5 * c.M_earth.cgs.value * (Zfil/ 0.02)**(1/2) * (gamma / np.pi**(-1))**(3/2) * (h / 0.05)**3 * (MS / c.M_sun.cgs.value)
        membryo = 10 * mp * pf
        membryo = np.minimum(self.Mvortex, membryo)
        return membryo
    
    def embryoMass(self, pf):
        """
        Calculates the embryo mass using eq.X and assuming
        that the embryo mass is 10x the standard mass of
        a planetesimal forming via the SI
        """
        gamma = 1
        Zfil = 0.1
        Omega = (c.G.cgs.value * MS / self.model.r**3)**(1/2)
        Cs = (c.k_B.cgs.value * self.model.temp / (mu * c.u.cgs.value))**(1/2)
        H = Cs / Omega[None,:]
        h = H / self.model.r[None,:]
        mp = 5e-5 * c.M_earth.cgs.value * (Zfil/ 0.02)**(1/2) * (gamma / np.pi**(-1))**(3/2) * (h / 0.05)**3 * (MS / c.M_sun.cgs.value)
        membryo = 10 * mp * pf
        return membryo
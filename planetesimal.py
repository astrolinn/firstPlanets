import numpy as np
import astropy.constants as c
from inputFile import (
    MS,mu
)

class Planetesimal:

    def __init__(self, model, d2gSt):
        self.model = model
        self.d2gSt = d2gSt
        self.rho_R = self.rocheDensity()
    
    def rocheDensity(self):
        G = c.G.cgs.value
        Omega = np.sqrt(G*MS/self.model.r**3)
        rho_R = 9*Omega**2 / (4*np.pi*G)
        return rho_R

    def SI_Lim2024b(self, St):
        eps_crit = 10**( 0.42 * np.log10(St)**2 + 0.72 * np.log10(St) + 0.37 )
        return eps_crit

    def planForm(self, eps, eps_crit):
        # Check if SI criteria or Roche density criteria is met 
        pf = np.array((eps > eps_crit) | (eps * self.model.rho_gas > self.rho_R[None,:]), dtype=int)
        return pf

    # Calculate the embryo mass
    def embryoMass(self, pf):
        gamma = 1
        Zfil = 0.1
        ME = c.M_earth.cgs.value
        kB = c.k_B.cgs.value
        mH = c.u.cgs.value
        G = c.G.cgs.value
        Omega = (G * MS / self.model.r**3)**(1/2)
        Cs = (kB * self.model.temp / (mu * mH))**(1/2)
        Hg = Cs / Omega[None,:]
        h = Hg / self.model.r[None,:]
        mp = 5e-5 * ME * (Zfil/ 0.02)**(1/2) * (gamma / np.pi**(-1))**(3/2) * (h / 0.05)**3 * (MS / c.M_sun.cgs.value)
        membryo = 10 * mp * pf
        return membryo
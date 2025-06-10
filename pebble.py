import astropy.constants as c
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import i0,i1

import input_loader

class Pebble:
    """
    This class handles the accretion of pebbles onto a planet
    inside and outside a vortex
    """
    def __init__(self, model, pfEm):
        self.model = model
        input_file = input_loader.load_input_file(self.model.path)
        self.dt_pebb = input_file.dt_pebb
        self.tend = input_file.tend
        self.alphaTurb = input_file.alphaTurb
        self.Mstar = input_file.Mstar
        self.mu = input_file.mu
        self.fracmax = input_file.fracmax
        self.lim3d = input_file.lim3d
        self.pfEm = pfEm

    def interp_form(self, tform, rform):
        """
        Finds the indices on the dustPy grid closest to
        the formation time and semimajor axis of the embryo
        """
        tind = np.abs(self.model.t - tform).argmin()
        rind = np.abs(self.model.r - rform).argmin()
        return tind, rind

    def timeArray(self, tstart, tfinish, timestep):
        """
        Initializes the time-array used for pebble accretion
        """
        num_points = int((tfinish - tstart) / timestep) + 1
        t = np.linspace(tstart, tfinish, num_points)
        return t
    
    def interp_t(self,t,array_model):
        """
        Interpolates between the time-array used for pebble
        accretion and the time-array that the dustPy data is
        saved on
        """
        array_pebb = interp1d(self.model.t, array_model, kind='linear', axis=0, fill_value='extrapolate')(t)
        return array_pebb

    def calc_dlnPdlnr(self):
        """
        Calculates the radial pressure gradient along the
        entire time and semimajor axis grid
        """
        r = self.model.r
        Omega = np.sqrt(c.G.cgs.value * self.Mstar / r**3)
        Cs = np.sqrt(c.k_B.cgs.value * self.model.temp / (self.mu * c.u.cgs.value))
        H = Cs / Omega
        dlnsigma_gasdlnr = np.gradient(np.log(self.model.sigma_gas), axis=1) / np.gradient(np.log(r))
        dlntempdlnr = np.gradient(np.log(self.model.temp), axis=1) / np.gradient(np.log(r))
        dlnHdlnr = np.gradient(np.log(H), axis=1) / np.gradient(np.log(r))
        dlnPdlnr = dlnsigma_gasdlnr + dlntempdlnr - dlnHdlnr
        return dlnPdlnr
    
    def calc_Miso(self, a, temp, dlnPdlnr):
        """
        Calculates the pebble isolation mass at the formation
        time of the embryo
        """
        alpha3 = 0.001
        Omega = np.sqrt(c.G.cgs.value * self.Mstar / a**3)
        Cs = np.sqrt(c.k_B.cgs.value * temp / (self.mu * c.u.cgs.value))
        H = Cs / Omega
        ffit = (H / a / 0.05)**3 * (0.34 * (np.log(alpha3) / np.log(self.alphaTurb))**4 + 0.66) * (1 - (dlnPdlnr + 2.5) / 6)
        return 25 * c.M_earth.cgs.value * ffit
    
    def pebbAccRate(self, a, temp, m, St_poly, sigma_dust_poly, rho_dust_poly, dlnPdlnr):
        """
        Calculates the pebble accretion rate using eq.35
        of Lyra et al. (2023) for the case of polydisperse
        pebble accretion
        The standard 2D limit (eq.30 of Lyra et al. 2023)
        is used as well to avoid numerical issues
        DustPy produces dust fluids with St>>1, to avoid
        numerical issues we remove the high tail of the
        stokes distribution
        Note: the focusing regime was not included in this version
        """
        chi = 0.4
        gamma = 0.65
        Omega = np.sqrt(c.G.cgs.value * self.Mstar / a**3)
        Cs = np.sqrt(c.k_B.cgs.value * temp / (self.mu * c.u.cgs.value))
        H = Cs / Omega
        Deltav = -0.5 * H / a * dlnPdlnr * Cs
        rH = (m / (3 * self.Mstar))**(1/3) * a
        tp = c.G.cgs.value * m / (Deltav + Omega * rH)**3
        Mt = Deltav**3 / (c.G.cgs.value * Omega)
        indmax = np.argmax(np.where(np.cumsum(rho_dust_poly, -1) / np.sum(rho_dust_poly, -1) < self.fracmax, St_poly, 0.0)) + 1
        Mdot_core = np.zeros((indmax))
        for i in range(indmax):
            Hd = H * np.sqrt(self.alphaTurb / (self.alphaTurb + St_poly[i]))
            tau_f = St_poly[i] / Omega
            M_HB = Mt / (8 * St_poly[i])
            if (m < M_HB):
                # Bondi regime
                R_B = c.G.cgs.value * m / Deltav**2
                t_B = R_B / Deltav
                Racc_hat = (4 * tau_f / t_B)**0.5 * R_B
            else:
                # Hill regime
                Racc_hat = (St_poly[i] / 0.1)**(1/3) * rH
            Racc = Racc_hat * np.exp(-chi * (tau_f / tp)**gamma)
            deltav = Deltav + Omega * Racc
            eps = (Racc / (2 * Hd))**2
            if (eps < self.lim3d):
                I0 = i0(eps)
                I1 = i1(eps)
                Mdot_core[i] = np.pi * Racc**2 * rho_dust_poly[i] * deltav * np.exp(-eps) * (I0+I1)
            else:
                Mdot_core[i] = 2 * Racc * sigma_dust_poly[i] * deltav
        Mdot_core_tot = sum(Mdot_core)
        return Mdot_core_tot
    
    def pebbAcc(self, tform, rform, mform):
        """
        Solves for the mass growth of a single planet via
        pebble accretion
        """
        tind, rind = self.interp_form(tform, rform)
        dlnPdlnr = self.calc_dlnPdlnr()
        Miso = self.calc_Miso(self.model.r[rind], self.model.temp[tind,rind], dlnPdlnr[tind,rind])
        t = self.timeArray(tform, self.tend, self.dt_pebb)
        mPlanet = np.zeros((len(t)))
        mPlanet[0] = mform
        St_poly = self.interp_t(t, self.model.St_poly[:,rind,:])
        sigma_dust_poly = self.interp_t(t, self.model.sigma_dust_poly[:,rind,:])
        rho_dust_poly = self.interp_t(t, self.model.rho_dust_poly[:,rind,:])
        dlnPdlnr= self.interp_t(t, dlnPdlnr[:,rind])
        temp = self.interp_t(t, self.model.temp[:,rind])
        a = self.model.r[rind]
        for it in range(1, len(t)):
            m = mPlanet[it-1]
            if (m < Miso):
                Mdot_core_planet = self.pebbAccRate(a, temp[it], m, St_poly[it,:], sigma_dust_poly[it,:], rho_dust_poly[it,:], dlnPdlnr[it])
                mnew = m + Mdot_core_planet * self.dt_pebb
            else:
                mnew = m
            mPlanet[it] = mnew
        return mPlanet[-1]
    
    def pebbAcc_vortex_v2(self, tform, rform, mform, tLeaveVortex):
        """
        Solves for the mass growth of a planet inside a vortex
        via the accretion of pebbles, from time tform to time
        tLeaveVortex
        1) The planet accretes all dust that is trapped within
        the vortex during the first orbital period (if the planet
        leaves the vortex after e.g. 0.6 orbital periods it 
        accretes 60% of the trapped dust mass)
        2) For the remainder of the time within the vortex, the 
        planet accretes at the rate which dust radially drifts
        into the vortex semimajor axis (which is ~constant)
        """
        tind, rind = self.interp_form(tform, rform)
        orbitalPeriod = 2*np.pi * np.sqrt(rform**3/(c.G.cgs.value*self.Mstar))
        dttot = tLeaveVortex - tform
        dt1 = min(dttot, orbitalPeriod)
        dt2 = dttot - dt1
        Mvortex_dust = self.pfEm.Mvortex[tind, rind] - mform
        mPlanet = mform + Mvortex_dust * dt1 / orbitalPeriod
        MdotIntoVortex_poly = self.model.sigma_dust_poly[tind,rind,:] * 2 * np.pi * rform * self.model.vrad_dust_poly[tind,rind,:]
        MdotIntoVortex = MdotIntoVortex_poly.sum()
        mPlanet = mPlanet + MdotIntoVortex * dt2
        return mPlanet

    ############## Old functions no longer in use ################

    def pebbAccRate_vortex(self, tind, rind):
        """
        Calculates the pebble accretion rate inside a vortex
        using eq.27 from Cummins et al. (2022)
        """
        a = self.model.r[rind]
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        temp = self.model.temp[tind,rind]
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega
        Mdot_core = 4/3 * Omega * H**2 * self.model.sigma_dust_poly[tind,rind,:].sum(-1)
        return Mdot_core
    
    def pebbAcc_vortex(self, tform, rform, mform, tLeaveVortex):
        """
        Solves for the mass growth of a planet inside a vortex
        via the accretion of pebbles, from time tform to time
        tLeaveVortex
        1) The planet accretes all dust that is trapped within
        the vortex during the first orbital period (if the planet
        leaves the vortex after e.g. 0.6 orbital periods it 
        accretes 60% of the trapped dust mass)
        2) For the remainder of the time within the vortex, the 
        planet accretes at a rate set by eq.27 from Cummins et al. 
        (whic is constant because the surface density of the 
        background disk is ~constant)
        """
        tind, rind = self.interp_form(tform, rform)
        orbitalPeriod = 2*np.pi * np.sqrt(rform**3/(c.G.cgs.value*MS))
        dttot = tLeaveVortex - tform
        dt1 = min(dttot, orbitalPeriod)
        dt2 = dttot - dt1
        Mvortex_dust = self.pfEm.Mvortex[tind, rind] - mform
        mPlanet = mform + Mvortex_dust * dt1 / orbitalPeriod
        mPlanet = mPlanet + self.pebbAccRate_vortex(tind, rind) * dt2
        return mPlanet

    def areaVortex(self, tind, rind):
        """
        Calculates the area of a Kida vortex with chi=4
        """
        chi = 4
        a = self.model.r[rind]
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        temp = self.model.temp[tind,rind]
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega
        area = 4/9 * np.pi * chi * H**2
        return area
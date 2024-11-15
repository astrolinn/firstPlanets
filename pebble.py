import numpy as np
import astropy.constants as c
from scipy.interpolate import interp1d
from scipy.special import i0,i1
from inputFile import (
    dt, tend, alphaTurb, sigmaExp, tempExp, 
    HExp, MS, mu, fracmax, lim3d
)

class Pebble:
    """
    This class handles the accretion of pebbles onto a planet
    inside and outside a vortex
    """
    def __init__(self, model, pfEm):
        self.model = model
        self.pfEm = pfEm

    def interp_form(self, tform, rform):
        """
        Finds the indices in the full dustPy data closest to
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
    
    def calc_Miso(self,tind,rind):
        """
        Calculates the pebble isolation mass at the formation
        time of the embryo
        """
        alpha3 = 0.001
        a = self.model.r[rind]
        temp = self.model.temp[tind,rind]
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega
        dlnPdlnR = (-sigmaExp) + (-tempExp) - HExp
        ffit = (H/a/0.05)**3 * (0.34*(np.log(alpha3)/np.log(alphaTurb))**4+0.66) * (1-(dlnPdlnR + 2.5)/6)
        return 25 * c.M_earth.cgs.value * ffit
    
    def pebbAccRate(self, m, St_poly, sigma_dust_poly, rho_dust_poly, tind, rind):
        """
        Calculates the pebble accretion rate using eq.35 
        of Lyra et al. (2023) for the case of polydisperse
        pebble accretion
        The standard 2D limit (eq.30 of Lyra et al. 2023)
        is used as well to avoid numerical issues
        DustPy produces dust fluids with St>>1, to avoid
        numerical issues we remove the high tail of the
        stokes distribution
        """
        a = self.model.r[rind]
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        temp = self.model.temp[tind,rind]
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega
        rH = (m/(3*MS))**(1/3)*a
        indmax = np.argmax(np.where(np.cumsum(rho_dust_poly,-1)/np.sum(rho_dust_poly,-1)<fracmax, St_poly, 0.0)) + 1
        Mdot_core = np.zeros((indmax))
        for i in range(indmax):
            Hd = H * np.sqrt(alphaTurb/(alphaTurb+St_poly[i]))
            Racc = (St_poly[i]/0.1)**(1/3)*rH
            deltav = Omega*Racc
            eps = (Racc/(2*Hd))**2
            if (eps<lim3d):
                I0 = i0(eps)
                I1 = i1(eps)
                Mdot_core[i] = np.pi*Racc**2*rho_dust_poly[i]*deltav * np.exp(-eps) * (I0+I1)
            else:
                Mdot_core[i] = 2*Racc*sigma_dust_poly[i]*deltav
        Mdot_core_tot = sum(Mdot_core)
        return Mdot_core_tot
    
    def pebbAcc(self, tform, rform, mform):
        """
        Solves for the mass growth of a single planet via
        pebble accretion
        """
        tind, rind = self.interp_form(tform, rform)
        Miso = self.calc_Miso(tind,rind)
        t = self.timeArray(tform, tend, dt)
        mPlanet = np.zeros((len(t)))
        mPlanet[0] = mform
        St_poly = self.interp_t(t,self.model.St_poly[:,rind,:])
        sigma_dust_poly = self.interp_t(t,self.model.sigma_dust_poly[:,rind,:])
        rho_dust_poly = self.interp_t(t,self.model.rho_dust_poly[:,rind,:])
        for it in range(1,len(t)):
            m = mPlanet[it-1]
            if (m<Miso):
                Mdot_core_planet = self.pebbAccRate(m,St_poly[it,:],sigma_dust_poly[it,:],rho_dust_poly[it,:],tind,rind)
                mnew = m + Mdot_core_planet * dt
            else:
                mnew = m
            mPlanet[it] = mnew
        return mPlanet[-1]

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
        orbitalPeriod = 2*np.pi * np.sqrt(rform**3/(c.G.cgs.value*MS))
        dttot = tLeaveVortex - tform
        dt1 = min(dttot, orbitalPeriod)
        dt2 = dttot - dt1
        Mvortex_dust = self.pfEm.Mvortex[tind, rind] - mform
        mPlanet = mform + Mvortex_dust * dt1 / orbitalPeriod
        MdotIntoVortex_poly = self.model.sigma_dust_poly[tind,rind,:] * 2 * np.pi * rform * self.model.vrad_dust_poly[tind,rind,:]
        MdotIntoVortex = MdotIntoVortex_poly.sum()
        mPlanet = mPlanet + MdotIntoVortex * dt2
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

    def pebbAccRate_vortex_oldversion(self, tind, rind, sigma_dust_vortex):
        """
        Calculates the pebble accretion rate inside a vortex
        using eq.27 from Cummins et al. (2022)
        """
        a = self.model.r[rind]
        Omega = np.sqrt(c.G.cgs.value*MS/a**3)
        temp = self.model.temp[tind,rind]
        Cs = np.sqrt(c.k_B.cgs.value*temp/(mu*c.u.cgs.value))
        H = Cs/Omega
        Mdot_core = 4/3 * Omega * H**2 * sigma_dust_vortex
        return Mdot_core

    def pebbAcc_vortex_oldversion(self, tform, rform, mform, tLeaveVortex):
        """
        Solves for the mass growth of a planet inside a vortex
        via the accretion of pebbles, from time tform to time
        tLeaveVortex
        The pebble accretion rate is limited to 50% of the
        total dust mass in the vortex, which changes with 
        time as pebbles are accreted and new pebbles drift 
        into the vortex

            MdotIntoVortex: mass flux of pebbles drifting into
        the vortex from further out in the disk
        """
        tind, rind = self.interp_form(tform, rform)
        orbitalPeriod = 2*np.pi * np.sqrt(rform**3/(c.G.cgs.value*MS))
        dt_vortex = orbitalPeriod/10
        t = self.timeArray(tform, tLeaveVortex, dt_vortex)
        mPlanet = np.zeros((len(t)))
        mPlanet[0] = mform
        MdotIntoVortex_poly = self.model.sigma_dust_poly[tind,rind,:] * 2 * np.pi * rform * self.model.vrad_dust_poly[tind,rind,:]
        MdotIntoVortex = MdotIntoVortex_poly.sum()
        AreaVortex = self.areaVortex(tind, rind)
        Mvortex = self.pfEm.Mvortex[tind, rind]
        for it in range(1,len(t)):
            m = mPlanet[it-1]
            sigma_dust_vortex = Mvortex/AreaVortex
            Mdot_core_planet = self.pebbAccRate_vortex(tind, rind, sigma_dust_vortex)
            mnew = m + min(0.5 * Mvortex, Mdot_core_planet*dt_vortex)
            mPlanet[it] = mnew
            Mvortex = Mvortex - Mdot_core_planet * dt_vortex + (-MdotIntoVortex) * dt_vortex
        return mPlanet[-1]
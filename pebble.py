import numpy as np
import astropy.constants as c
from scipy.interpolate import interp1d
from inputFile import (
    dt, tend, alphaTurb, Mdot_core_min
)

class Pebble:

    def __init__(self, model, tform, rform):
        self.model = model
        self.tform = tform
        self.rform = rform

    def interp_form(self):
        tind = np.abs(self.model.t - self.tform).argmin()
        rind = np.abs(self.model.r - self.rform).argmin()
        return tind, rind
    
    def timeArray(self):
        t = np.linspace(self.tform,tend,int(np.floor(tend/dt)))
        return t
    
    def interp_t(self,t,array_model):
        array_pebb = interp1d(self.model.t, array_model, kind='linear', axis=0, fill_value='extrapolate')(t)
        return array_pebb
    
    def calc_Miso(self,tind,rind):
        temp = self.model.temp[tind,rind]
        Miso = 0
        return Miso
    
    def pebbAccRate(self, m, st_poly, sigma_dust_poly, rho_dust_poly):
        pass
    
    def pebbAcc(self, membryo, membryo_vortex, membryo_vortex_growth):
        tind, rind = self.interp_form()
        me = membryo[tind,rind]
        me_v = membryo_vortex[tind,rind]
        me_v_g = membryo_vortex_growth[tind,rind]
        if me > 0 or me_v > 0 or me_v_g > 0:
            mform = max(me,me_v,me_v_g)
            # Calc Pebble Isolation Mass
            Miso = self.calc_Miso(tind,rind)
            # Initiate time array for planet formation, and array to store the planetary mass
            t = self.timeArray()
            mPlanet = np.zeros((len(t)))
            mPlanet[0] = mform
            # Interpolate to obtain stokes and dust densities on time array for planet formation
            st_poly = self.interp_t(t,self.model.st_poly[:,rind,:])
            sigma_dust_poly = self.interp_t(t,self.model.sigma_dust_poly[:,rind,:])
            rho_dust_poly = self.interp_t(t,self.model.rho_dust_poly[:,rind,:])
            # Perform pebble accretion onto the planet
            for it in range(1,len(t)):
                m = mPlanet[it-1]
                if (m<Miso):
                    Mdot_core_planet = self.pebbAccRate(m,st_poly[it,:],sigma_dust_poly[it,:],rho_dust_poly[it,:])
                    Mdot_core_planet = max(Mdot_core_planet,Mdot_core_min)
                    mnew = m + Mdot_core_planet * dt
                else:
                    Mdot_core_planet = Mdot_core_min
                    mnew = m + Mdot_core_planet * dt
                mPlanet[it] = mnew

        if me > 0:
            mp = mPlanet[-1]
        if me_v > 0:
            mp_v = mPlanet[-1]
        if me_v_g > 0:
            mp_v_g = mPlanet[-1]
        return mp, mp_v, mp_v_g
    
    def pebbAcc_vortex(self, membryo, membryo_vortex, membryo_vortex_growth, orbInVortex=10):
        tind, rind = self.interp_form()
        me = membryo[tind,rind]
        me_v = membryo_vortex[tind,rind]
        me_v_g = membryo_vortex_growth[tind,rind]
        if me > 0 or me_v > 0 or me_v_g > 0:
            mform = max(me,me_v,me_v_g)
            # Calc Pebble Isolation Mass
            Miso = self.calc_Miso(tind,rind)
            # Initiate time array for planet formation, and array to store the planetary mass
            t = self.timeArray()
            mPlanet = np.zeros((len(t)))
            mPlanet[0] = mform
            # Interpolate to obtain stokes and dust densities on time array for planet formation
            st_poly = self.interp_t(t,self.model.st_poly[:,rind,:])
            sigma_dust_poly = self.interp_t(t,self.model.sigma_dust_poly[:,rind,:])
            rho_dust_poly = self.interp_t(t,self.model.rho_dust_poly[:,rind,:])
            # Perform pebble accretion onto the planet
            for it in range(1,len(t)):
                m = mPlanet[it-1]
                if (m<Miso):
                    Mdot_core_planet = self.pebbAccRate(m,st_poly[it,:],sigma_dust_poly[it,:],rho_dust_poly[it,:])
                    Mdot_core_planet = max(Mdot_core_planet,Mdot_core_min)
                    mnew = m + Mdot_core_planet * dt
                else:
                    Mdot_core_planet = Mdot_core_min
                    mnew = m + Mdot_core_planet * dt
                mPlanet[it] = mnew

        if me > 0:
            mp = mPlanet[-1]
        if me_v > 0:
            mp_v = mPlanet[-1]
        if me_v_g > 0:
            mp_v_g = mPlanet[-1]
        return mp, mp_v, mp_v_g
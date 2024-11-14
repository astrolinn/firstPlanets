import numpy as np

import glob

class Model:
    """
    This class handles reading in the dustPy data
    """

    def __init__(self, tmin, rmax):
        self.tmin = tmin
        self.rmax = rmax
        self.load()
        self.limit()
    
    def load(self):
        self.r = np.load('files_dp/r.npy')
        self.t = np.load('files_dp/tdustev.npy')
        self.sigma_gas = np.load('files_dp/sigma_gas_2D.npy') # (t,r)
        self.sigma_dust_poly = np.load('files_dp/sigma_dust_poly_3D.npy') # (t,r,fluid-bins)
        self.rho_gas = np.load('files_dp/rho_gas_2D.npy') # (t,r)
        self.rho_dust_poly = np.load('files_dp/rho_dust_3D.npy') # (t,r,fluid-bins)
        self.St_poly = np.load('files_dp/st_poly_3D.npy') # (t,r,fluid-bins)
        self.temp = np.load('files_dp/temp_2D.npy') # (t,r)
        self.vrad_dust_poly = np.load('files_dp/vrad_dust_3D.npy') # (t,r,fluid-bins)

    def limit(self):
        """
        Remove data at semimajor axes beyond rmax and
        time before tlim
        """
        ind_t = np.argmax(self.t > self.tmin)
        ind_r = np.argmax(self.r > self.rmax)
        self.t = self.t[ind_t:]
        self.r = self.r[0:ind_r]
        self.sigma_gas = self.sigma_gas[ind_t:,0:ind_r]
        self.sigma_dust_poly = self.sigma_dust_poly[ind_t:,0:ind_r,:]
        self.rho_gas = self.rho_gas[ind_t:,0:ind_r]
        self.rho_dust_poly = self.rho_dust_poly[ind_t:,0:ind_r,:]
        self.St_poly = self.St_poly[ind_t:,0:ind_r,:]
        self.temp = self.temp[ind_t:,0:ind_r]
        self.vrad_dust_poly = self.vrad_dust_poly[ind_t:,0:ind_r,:]




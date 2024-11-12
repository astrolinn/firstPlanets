import numpy as np

import glob

class Model:

    path = '/Users/linn/planFormCode/firstPlanets/pythonFiles_vf1/'

    def __init__(self, frag_vel, dust2gas, alpha_turb, tmin, rmax):
        self.frag_vel = frag_vel
        self.dust2gas = dust2gas
        self.alpha_turb = alpha_turb
        self.tmin = tmin
        self.rmax = rmax

        self.model_path = self.path + "vf{0}_Z{1}_at{2}/files_dp/".format(
            self.frag_vel, self.dust2gas, self.alpha_turb)

        self.load()
        self.limit()
    
    def load(self):
        self.r = np.load(self.model_path+'r.npy')
        self.t = np.load(self.model_path+'tdustev.npy')

        self.sigma_gas = np.load(self.model_path+'sigma_gas_2D.npy') # (t,r)
        self.sigma_dust_poly = np.load(self.model_path+'sigma_dust_poly_3D.npy') # (t,r,bins)
        self.rho_gas = np.load(self.model_path+'rho_gas_2D.npy') # (t,r)
        self.rho_dust_poly = np.load(self.model_path+'rho_dust_3D.npy') # (t,r,bins)
        self.st_poly = np.load(self.model_path+'st_poly_3D.npy') # (t,r,bins)
        self.temp = np.load(self.model_path+'temp_2D.npy') # (t,r)

    def limit(self):
        ind_t = np.argmax(self.t > self.tmin)
        ind_r = np.argmax(self.r > self.rmax)
        self.t = self.t[ind_t:]
        self.r = self.r[0:ind_r]

        self.sigma_gas = self.sigma_gas[ind_t:,0:ind_r]
        self.sigma_dust_poly = self.sigma_dust_poly[ind_t:,0:ind_r,:]
        self.rho_gas = self.rho_gas[ind_t:,0:ind_r]
        self.rho_dust_poly = self.rho_dust_poly[ind_t:,0:ind_r,:]
        self.st_poly = self.st_poly[ind_t:,0:ind_r,:]
        self.temp = self.temp[ind_t:,0:ind_r]

    def interpolate(self, t, at_r_value):
        pass




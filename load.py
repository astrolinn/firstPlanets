import numpy as np
import os
import re

import glob
import warnings

class Model:
    """
    This class handles reading in the dustPy data and saving it
    as numpy arrays.
    """

    def __init__(self, path, tmin, rmax):
        self.path = path
        self.tmin = tmin
        self.rmax = rmax
        self.load()
        self.limit()
    
    def load(self):
        if os.path.exists(os.path.join(self.path, 'files_dp')):
            print("files_dp exists: loading data")
            self.r = np.load(os.path.join(self.path, 'files_dp/r.npy'))
            self.t = np.load(os.path.join(self.path, 'files_dp/tdustev.npy'))
            self.sigma_gas = np.load(os.path.join(self.path, 'files_dp/sigma_gas_2D.npy')) # (t,r)
            self.sigma_dust_poly = np.load(os.path.join(self.path, 'files_dp/sigma_dust_3D.npy')) # (t,r,fluid-bins)
            self.rho_gas = np.load(os.path.join(self.path, 'files_dp/rho_gas_2D.npy')) # (t,r)
            self.rho_dust_poly = np.load(os.path.join(self.path, 'files_dp/rho_dust_3D.npy')) # (t,r,fluid-bins)
            self.St_poly = np.load(os.path.join(self.path, 'files_dp/st_3D.npy')) # (t,r,fluid-bins)
            self.temp = np.load(os.path.join(self.path, 'files_dp/temp_2D.npy')) # (t,r)
            self.vrad_dust_poly = np.load(os.path.join(self.path, 'files_dp/vrad_dust_3D.npy')) # (t,r,fluid-bins)
            print("Data has been loaded")
        elif os.path.exists(os.path.join(self.path, 'data')):
            print("data directory exists: reading and saving dustPy data")
            import dustpy
            from dustpy import hdf5writer
            wrtr = hdf5writer()
            lent = self.findnrsave() + 1
            data = wrtr.read.output(os.path.join(self.path, 'data', 'data0000.hdf5'))
            nrfluids = len(data.dust.Sigma[0,:])
            self.r = data.grid.r
            self.t = np.zeros((lent))
            self.sigma_gas = np.zeros((lent, len(self.r)))
            self.sigma_dust_poly = np.zeros((lent, len(self.r), nrfluids))
            self.rho_gas = np.zeros((lent,len(self.r)))
            self.rho_dust_poly = np.zeros((lent, len(self.r), nrfluids))
            self.vrad_dust_poly = np.zeros((lent, len(self.r), nrfluids))
            self.temp = np.zeros((lent,len(self.r)))
            self.St_poly = np.zeros((lent, len(self.r), nrfluids))
            for i in range(lent):
                filename = f"data{i:04d}.hdf5"
                data = wrtr.read.output(os.path.join(self.path, 'data', filename))
                self.t[i] = data.t
                self.sigma_gas[i,:] = data.gas.Sigma
                self.sigma_dust_poly[i,:,:] = data.dust.Sigma
                self.rho_gas[i,:] = data.gas.rho
                self.rho_dust_poly[i,:,:] = data.dust.rho
                self.vrad_dust_poly[i,:,:] = data.dust.v.rad
                self.temp[i,:] = data.gas.T
                self.St_poly[i,:,:] = data.dust.St
            os.mkdir(self.path+'/files_dp')
            np.save(os.path.join(self.path, 'files_dp/r.npy'), self.r)
            np.save(os.path.join(self.path, 'files_dp/tdustev.npy'), self.t)
            np.save(os.path.join(self.path, 'files_dp/sigma_gas_2D.npy'), self.sigma_gas)
            np.save(os.path.join(self.path, 'files_dp/sigma_dust_3D.npy'), self.sigma_dust_poly)
            np.save(os.path.join(self.path, 'files_dp/rho_gas_2D.npy'), self.rho_gas)
            np.save(os.path.join(self.path, 'files_dp/rho_dust_3D.npy'), self.rho_dust_poly)
            np.save(os.path.join(self.path, 'files_dp/vrad_dust_3D.npy'), self.vrad_dust_poly)
            np.save(os.path.join(self.path, 'files_dp/temp_2D.npy'), self.temp)
            np.save(os.path.join(self.path, 'files_dp/st_3D.npy'), self.St_poly)
            print("Data has been loaded")
        else:
            raise FileNotFoundError("Error: no data exists in the specified path.")

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

    def findnrsave(self):
        """
        Find nr of saved snapshots from dustPy
        """
        directory = os.path.join(self.path, 'data')
        files = os.listdir(directory)
        pattern = re.compile(r'data(\d+)\.hdf5')
        indices = []
        for filename in files:
            match = pattern.match(filename)
            if match:
                indices.append(int(match.group(1)))
        if indices:
            max_index = max(indices)
        return max_index

import astropy.constants as c
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

year = 365.25*24*3600
au = c.au.cgs.value
ME = c.M_earth.cgs.value

class Plot:
    """
    This class contains functions for creating plots and
    saving numpy arrays
    """
    def __init__(self, model, d2gSt, path):
        self.path = path
        if not os.path.exists(os.path.join(self.path, 'plots')):
            os.makedirs(os.path.join(self.path, 'plots'))
        self.model = model
        self.d2gSt = d2gSt

    def plotDisk(self, membryo, membryo_vortex, membryo_vortex_growth, membryo_a, membryo_vortex_a, membryo_vortex_growth_a):
        """
        Plots the representative Stokes number, dust-to-gas
        midplane density ratio and embryo mass (where
        planetesimals form) as a function of time and
        semimajor axis, for the following three cases:
        1) No vortices
        2) Vortices, dust concentration (case VC)
        3) Vortices, concentration and dust growth (case VCG)
        Saves the relevant numpy arrays
        """
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=[10,9], sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)
        # Stokes
        Stmin = 1e-4
        Stmax = 1
        axes[0,0].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.St_mono, norm=LogNorm(vmin=Stmin,vmax=Stmax))
        axes[0,1].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.St_mono, norm=LogNorm(vmin=Stmin,vmax=Stmax))
        im0=axes[0,2].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.St_vortex_growth, norm=LogNorm(vmin=Stmin,vmax=Stmax))
        cbar0 = fig.colorbar(im0, ax=axes[0, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar0.set_label('St')
        # Dust-to-gas ratios
        d2gmin = 5e-4
        d2gmax = 5e3
        axes[1,0].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.d2gRho, norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        axes[1,1].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.d2gRho_vortex, norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        im1=axes[1,2].pcolormesh(self.model.r/au, self.model.t/year, self.d2gSt.d2gRho_vortex_growth, norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        cbar1 = fig.colorbar(im1, ax=axes[1, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar1.set_label(r'$\epsilon$')
        # Embryo masses
        emin = 0
        emax = 0.03
        axes[2,0].pcolormesh(self.model.r/au, self.model.t/year, np.where(membryo == 0, np.nan, membryo / ME), vmin=emin, vmax=emax)
        axes[2,1].pcolormesh(self.model.r/au, self.model.t/year, np.where(membryo_vortex == 0, np.nan, membryo_vortex / ME), vmin=emin, vmax=emax)
        im2=axes[2,2].pcolormesh(self.model.r/au, self.model.t/year, np.where(membryo_vortex_growth == 0, np.nan, membryo_vortex_growth / ME), vmin=emin, vmax=emax)
        cbar2 = fig.colorbar(im2, ax=axes[2, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar2.set_label(r'$M_{\rm embryo}/\rm{M}_{\oplus}$')
        # Labels
        axes[1,0].set_ylabel('t [yr]')
        axes[2,1].set_xlabel('r [au]')
        axes[0,0].set_title('No vortex')
        axes[0,1].set_title('Conc. (VC)')
        axes[0,2].set_title('Conc. & grow (VCG)')
        
        plt.savefig(os.path.join(self.path, 'plots', 'disk.png'), bbox_inches='tight')
        #plt.show()
        plt.close()

        np.save(os.path.join(self.path, 'plots', 'r.npy'), self.model.r)
        np.save(os.path.join(self.path, 'plots', 't.npy'), self.model.t)
        np.save(os.path.join(self.path, 'plots', 'St_mono.npy'), self.d2gSt.St_mono)
        np.save(os.path.join(self.path, 'plots', 'St_vortex_growth.npy'), self.d2gSt.St_vortex_growth)
        np.save(os.path.join(self.path, 'plots', 'd2gRho.npy'), self.d2gSt.d2gRho)
        np.save(os.path.join(self.path, 'plots', 'd2gRho_vortex.npy'), self.d2gSt.d2gRho_vortex)
        np.save(os.path.join(self.path, 'plots', 'd2gRho_vortex_growth.npy'), self.d2gSt.d2gRho_vortex_growth)
        np.save(os.path.join(self.path, 'plots', 'membryo.npy'), membryo) # SI criteria b
        np.save(os.path.join(self.path, 'plots', 'membryo_vortex.npy'), membryo_vortex) # SI criteria b
        np.save(os.path.join(self.path, 'plots', 'membryo_vortex_growth.npy'), membryo_vortex_growth) # SI criteria b
        np.save(os.path.join(self.path, 'plots', 'membryo_a.npy'), membryo_a) # SI criteria a
        np.save(os.path.join(self.path, 'plots', 'membryo_vortex_a.npy'), membryo_vortex_a) # SI criteria a
        np.save(os.path.join(self.path, 'plots', 'membryo_vortex_growth_a.npy'), membryo_vortex_growth_a) # SI criteria a

    def plotPlanet(self, membryo_vortex, membryo_vortex_growth, tform_exact, rform_exact,
        mp_vortex, mp_vortex_growth, mp_vortex_2, mp_vortex_growth_2,
        mp_vortex_a, mp_vortex_growth_a, mp_vortex_2_a, mp_vortex_growth_2_a):
        """
        Plots the embryo mass, final planetary mass
        assuming no pebble accretion within the vortex,
        and final planetary mass assuming that the
        planet initially accretes pebbles within the
        vortex (where planetesimals form) as a function
        of time and semimajor axis, for case VC and VCG.
        Saves the related numpy arrays
        """
        fig, axes = plt.subplots(nrows=3,ncols=2,figsize=[7,9],sharex=True,sharey=True)
        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)
        # Embryo masses
        emin = 0
        emax = 0.03
        axes[0,0].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex == 0, np.nan, membryo_vortex / ME),vmin=emin,vmax=emax)
        im0=axes[0,1].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex_growth == 0, np.nan, membryo_vortex_growth / ME),vmin=emin,vmax=emax)
        cbar0 = fig.colorbar(im0, ax=axes[0, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar0.set_label(r'$M_{\rm embryo}/\rm{M}_{\oplus}$') 
        # Planetary masses when accretion is purely outside the vortex
        pmin = 0
        pmax = 0.03
        axes[1,0].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex == 0, np.nan, mp_vortex / ME),vmin=pmin,vmax=pmax)
        im1=axes[1,1].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_growth == 0, np.nan, mp_vortex_growth / ME),vmin=pmin,vmax=pmax)
        cbar1 = fig.colorbar(im1, ax=axes[1, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar1.set_label(r'$M_{\rm p}/\rm{M}_{\oplus}$ - 0P in vortex') 
        # Planetary masses when accretion is initially inside the vortex
        pmin = 0
        pmax = 0.3
        axes[2,0].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_2 == 0, np.nan, mp_vortex_2 / ME),vmin=pmin,vmax=pmax)
        im2=axes[2,1].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_growth_2 == 0, np.nan, mp_vortex_growth_2 / ME),vmin=pmin,vmax=pmax)
        cbar2 = fig.colorbar(im2, ax=axes[2, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar2.set_label(r'$M_{\rm p}/\rm{M}_{\oplus}$ - 10P in vortex') 
        # Labels
        axes[1,0].set_ylabel('t [yr]')
        axes[2,0].set_xlabel('r [au]') 
        axes[2,1].set_xlabel('r [au]') 
        axes[0,0].set_title('Conc. (VC)')
        axes[0,1].set_title('Conc. & grow (VCG)')
        axes[0,0].set_xlim([1,50])
        axes[0,0].set_ylim([0,3e6])
        plt.savefig(os.path.join(self.path,'planet.png'),bbox_inches='tight')
        plt.show()
        plt.close()
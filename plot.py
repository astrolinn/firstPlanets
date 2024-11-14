import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
year = 365.25*24*3600
au = c.au.cgs.value
ME = c.M_earth.cgs.value

class Plot:
    """
    This class contains functions for creating plots
    """
    def __init__(self, model, d2gSt):
        self.model = model
        self.d2gSt = d2gSt

    def plotDisk(self, membryo, membryo_vortex, membryo_vortex_growth):
        """
        Plots the representative Stokes number, dust-to-gas
        midplane density ratio and embryo mass (where 
        planetesimals form) as a function of time and
        semimajor axis, for the following three cases:
        1) No vortices
        2) Vortices everywhere, extra dust concentration
        3) Vortices everywhere, extra dust concentration,
        mass loading and dust growth
        """
        fig, axes = plt.subplots(nrows=3,ncols=3,figsize=[10,9],sharex=True,sharey=True)
        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)
        # Stokes
        Stmin = 1e-4
        Stmax = 1
        axes[0,0].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.St_mono,norm=LogNorm(vmin=Stmin,vmax=Stmax))
        axes[0,1].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.St_mono,norm=LogNorm(vmin=Stmin,vmax=Stmax))
        im0=axes[0,2].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.St_vortex_growth,norm=LogNorm(vmin=Stmin,vmax=Stmax))
        cbar0 = fig.colorbar(im0, ax=axes[0, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar0.set_label('St')
        # Dust-to-gas ratios
        d2gmin = 5e-4
        d2gmax = 5e3
        axes[1,0].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.d2gRho,norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        axes[1,1].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.d2gRho_vortex,norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        im1=axes[1,2].pcolormesh(self.model.r/au,self.model.t/year,self.d2gSt.d2gRho_vortex_growth,norm=LogNorm(vmin=d2gmin,vmax=d2gmax))
        cbar1 = fig.colorbar(im1, ax=axes[1, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar1.set_label(r'$\epsilon$')
        # Embryo masses
        emin = 0
        emax = 0.03
        axes[2,0].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo == 0, np.nan, membryo / ME),vmin=emin,vmax=emax)
        axes[2,1].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex == 0, np.nan, membryo_vortex / ME),vmin=emin,vmax=emax)
        im2=axes[2,2].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex_growth == 0, np.nan, membryo_vortex_growth / ME),vmin=emin,vmax=emax)
        cbar2 = fig.colorbar(im2, ax=axes[2, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar2.set_label(r'$M_{\rm embryo}/\rm{M}_{\oplus}$')  
        # Labels
        axes[1,0].set_ylabel('t [yr]')
        axes[2,1].set_xlabel('r [au]') 
        axes[0,0].set_title('No vortex')
        axes[0,1].set_title('Vortex - conc.')
        axes[0,2].set_title('Vortex - conc. & grow')
        plt.savefig("disk.png",bbox_inches="tight")
        plt.show()
        plt.close()

    def plotPlanet(self, membryo, membryo_vortex, membryo_vortex_growth, tform_exact, rform_exact, mp, mp_vortex, mp_vortex_growth, mp_2, mp_vortex_2, mp_vortex_growth_2):
        """
        Plots the embryo mass, final planetary mass 
        assuming no pebble accretion within the vortex, 
        and final planetary mass assuming that the 
        planet initially accretes pebbles within the 
        vortex (where planetesimals form) as a function 
        of time and semimajor axis, for the following 
        three cases:
        1) No vortices
        2) Vortices everywhere, extra dust concentration
        3) Vortices everywhere, extra dust concentration,
        mass loading and dust growth
        """
        fig, axes = plt.subplots(nrows=3,ncols=3,figsize=[10,9],sharex=True,sharey=True)
        plt.subplots_adjust(wspace=0.05)
        plt.subplots_adjust(hspace=0.05)
        # Embryo masses
        emin = 0
        emax = 0.03
        axes[0,0].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo == 0, np.nan, membryo / ME),vmin=emin,vmax=emax)
        axes[0,1].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex == 0, np.nan, membryo_vortex / ME),vmin=emin,vmax=emax)
        im0=axes[0,2].pcolormesh(self.model.r/au,self.model.t/year,np.where(membryo_vortex_growth == 0, np.nan, membryo_vortex_growth / ME),vmin=emin,vmax=emax)
        cbar0 = fig.colorbar(im0, ax=axes[0, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar0.set_label(r'$M_{\rm embryo}/\rm{M}_{\oplus}$') 
        # Planetary masses when accretion is purely outside the vortex
        pmin = 0
        pmax = 0.03
        axes[1,0].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp == 0, np.nan, mp / ME),vmin=pmin,vmax=pmax)
        axes[1,1].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex == 0, np.nan, mp_vortex / ME),vmin=pmin,vmax=pmax)
        im1=axes[1,2].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_growth == 0, np.nan, mp_vortex_growth / ME),vmin=pmin,vmax=pmax)
        cbar1 = fig.colorbar(im1, ax=axes[1, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar1.set_label(r'$M_{\rm p}/\rm{M}_{\oplus}$') 
        # Planetary masses when accretion is initially inside the vortex
        pmin = 0
        pmax = 0.25
        axes[2,0].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_2 == 0, np.nan, mp_2 / ME),vmin=pmin,vmax=pmax)
        axes[2,1].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_2 == 0, np.nan, mp_vortex_2 / ME),vmin=pmin,vmax=pmax)
        im2=axes[2,2].pcolormesh(rform_exact/au,tform_exact/year,np.where(mp_vortex_growth_2 == 0, np.nan, mp_vortex_growth_2 / ME),vmin=pmin,vmax=pmax)
        cbar2 = fig.colorbar(im2, ax=axes[2, :], orientation='vertical', fraction=0.2, pad=0.02)
        cbar2.set_label(r'$M_{\rm p}/\rm{M}_{\oplus}$ - vortex acc.') 
        # Labels
        axes[1,0].set_ylabel('t [yr]')
        axes[2,1].set_xlabel('r [au]') 
        axes[0,0].set_title('No vortex')
        axes[0,1].set_title('Vortex - conc.')
        axes[0,2].set_title('Vortex - conc. & grow')
        axes[0,0].set_xlim([1,50])
        axes[0,0].set_ylim([0,3e6])
        plt.savefig("planet.png",bbox_inches="tight")
        plt.show()
        plt.close()
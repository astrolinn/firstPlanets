"""
Script for initializing and running a dustpy simulation.

Classes:
    input_loader: load parameters from inputFile.py

Usage:
    Run this script from inside the firstPlanets directory
    to execute a dustpy simulation inside the simulation
    directory specified by path.

Example:
    $ python dustev.py --path pathToDir
"""

import argparse
from astropy import constants as c
import numpy as np

from dustpy import Simulation
from dustpy import std

import input_loader

au = c.au.cgs.value
G = c.G.cgs.value
kB = c.k_B.cgs.value
mH = c.u.cgs.value
sigma_sb = c.sigma_sb.cgs.value

def midplaneTemp(r,input):
    """
    Calculates the radial midplane temperature structure
    The choice of temperature profile is specified in inputFile.py
    """
    if input.tempProfile == "CG97":
        T = input.Tconst * (r/au)**(input.tempExp)
    elif input.tempProfile == "passIrr":
        Lstar = 4*np.pi*input.Rstar**2*sigma_sb*input.Tstar**4
        T = ( 1/2*0.05*Lstar/(4*np.pi*r**2*sigma_sb) )**(1/4)
    else:
        raise ValueError("Must choose temperature profile")
    return T

def viscAccDisc_grid(t,r,input):
    """
    Calculates the analytical gas surface density profile
    of a viscous accretion disk.
    Reference Lynden-Bell & Pringle (1974). See eq.1 of Eriksson et al. (2023).
    """
    Mdot0_r = np.ones((len(r)))*input.Mdot0
    gamma = 1.5 + input.tempExp
    Omega = np.sqrt(G*input.Mstar/input.Rout**3)
    Temp = midplaneTemp(input.Rout,input)
    Cs = np.sqrt(kB*Temp/(input.mu*mH))
    H = Cs/Omega
    nu_Rout = input.alpha*Omega*H**2
    t_s = 1.0/(3.0*(2.0-gamma)**2) * input.Rout**2/nu_Rout
    T_1 = t/t_s + 1
    if np.isscalar(T_1):
        Mdot = Mdot0_r * T_1**( -(5/2-gamma)/(2-gamma) )
        sigma = Mdot/(3*np.pi*nu_Rout*(r/input.Rout)**gamma) * np.exp(-(r/input.Rout)**(2-gamma)/T_1)
    else:
        Mdot = Mdot0_r * T_1[:, np.newaxis]**( -(5/2-gamma)/(2-gamma) )
        sigma = Mdot/(3*np.pi*nu_Rout*(r[np.newaxis, :]/input.Rout)**gamma) * np.exp(-(r[np.newaxis, :]/input.Rout)**(2-gamma)/T_1[:, np.newaxis])
    return Mdot,sigma

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to where inputFile.py is located.', default='./', type=str)
    args = parser.parse_args()

    input_file = input_loader.load_input_file(args.path)

    # Set up semimajor axis and time grid
    ri = np.geomspace(input_file.Rin_dust, input_file.Redge_dust, input_file.Rnr_dust+1)
    ri_outer = np.concatenate([np.arange(input_file.Rcoarse_int*(ri[-1]//input_file.Rcoarse_int+1.), input_file.Rgrid_out, input_file.Rcoarse_int), [input_file.Rgrid_out]])
    ri = np.concatenate([ri, ri_outer])
    t = np.linspace(input_file.tstart, input_file.tend, int(np.floor(input_file.tend/input_file.dt_dust)))

    ### Initialize dustPy
    sim = Simulation()
    ### Grid Configuration
    sim.ini.dust.rhoMonomer = input_file.rhop
    sim.ini.grid.Nmbpd = 7 # Default
    sim.ini.grid.mmin = 4./3. * np.pi * sim.ini.dust.rhoMonomer * input_file.dustMinSize**3
    sim.ini.grid.mmax = 4./3. * np.pi * sim.ini.dust.rhoMonomer * input_file.dustMaxSize**3
    sim.grid.ri = ri
    sim.makegrids()
    ### Stellar Parameters
    sim.ini.star.M = input_file.Mstar
    sim.ini.star.R = input_file.Rstar
    sim.ini.star.T = input_file.Tstar
    # Gas Parameters
    sim.ini.gas.alpha = input_file.alpha
    sim.ini.gas.gamma = 1.0 # Adiabatic index, set to 1 for isothermal
    sim.ini.gas.mu = input_file.mu*mH
    sim.ini.gas.SigmaExp = input_file.sigmaExp
    sim.ini.gas.SigmaRc = input_file.Rout
    ### Dust Parameters
    sim.ini.dust.aIniMax = input_file.a_0
    sim.ini.dust.allowDriftingParticles = input_file.allowDriftingParticles
    sim.ini.dust.d2gRatio = input_file.Z
    sim.ini.dust.vfrag = input_file.vfrag
    ### Initialize
    sim.initialize()
    ### Different dust diffusivity
    sim.dust.delta.rad[...] = input_file.alphaTurb
    sim.dust.delta.rad.updater = None
    sim.dust.delta.turb[...] = input_file.alphaTurb
    sim.dust.delta.turb.updater = None
    sim.dust.delta.vert[...] = input_file.alphaTurb
    sim.dust.delta.vert.updater = None
    sim.dust.update()
    ### Set the initial surface densities and temperature structure
    sim.gas.T[...] = midplaneTemp(sim.grid.r, input_file)
    sim.gas.T.updater = None
    sim.gas.update()
    Mdot_gas_0, sigma_gas_0 = viscAccDisc_grid(t[0], sim.grid.r, input_file)
    sim.gas.Sigma[...] = sigma_gas_0
    sim.gas.update()
    sim.dust.Sigma[...] = std.dust.MRN_distribution(sim)
    sim.dust.update()
    ### Update and finalize all fields
    sim.update()
    sim.integrator._finalize()
    ### Time between saved snapshots
    sim.t.snapshots = t
    ### Lots of things that are not saved
    sim.dust.kernel.save = False
    sim.dust.v.rel.azi.save = False
    sim.dust.v.rel.rad.save = False
    sim.dust.v.rel.brown.save = False
    sim.dust.v.rel.turb.save = False
    sim.dust.v.rel.vert.save = False
    sim.dust.Fi.adv.save = False
    sim.dust.Fi.diff.save = False
    sim.dust.coagulation.A.save = False
    sim.dust.coagulation.eps.save = False
    sim.dust.coagulation.phi.save = False
    sim.dust.coagulation.lf_ind.save = False
    sim.dust.coagulation.rm_ind.save = False
    sim.dust.coagulation.stick.save = False
    sim.dust.coagulation.stick_ind.save = False
    sim.dust.p.stick.save = False
    sim.dust.p.frag.save = False
    sim.dust.v.rel.tot.save = False
    ### Save statement
    sim.writer.datadir = args.path.rstrip('/') + '/data'
    sim.writer.overwrite = True
    ### Run dustPy
    sim.update()
    sim.run()

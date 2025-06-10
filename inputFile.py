# Contains all input parameters for the simulation
# Note: cgs units

from astropy import constants as c
import numpy as np
#########################################

year = 365.25*24*3600
au = c.au.cgs.value
Msolar = c.M_sun.cgs.value

#########################################

# Time-grid
tstart = 0          # Start point of time-grid
tend = 3e6*year     # End point of time-grid, also end of planet formation
dt_dust = 3e3*year  # Frequency at which outputs are saved from dust evolution
dt_pebb = 5e2*year  # Time-step for planet growth via pebble accretion

# Stellar parameters
Mstar= 0.45 * c.M_sun.cgs.value
Rstar  = 2.411 * c.R_sun.cgs.value
Tstar = 3761

# Protoplanetary disk parameters
Mdot0 = 1.32e-8 * Msolar/year
                    # Initial disk accretion rate (determines initial
                    # gas surface density)
Rout = 25.0*au      # Critical cut-off radius of surface density
Z = 0.0016          # Dust-to-gas surface density ratio
tempProfile = "passIrr"
                    # Choose between CG97 or passIrr
Tconst = 150.0      # Temperature at 1au in model CG97
tempExp = -0.5      # dlnT/dlnR
sigmaExp = -1.0     # dlnSigma/dlnR at 1au
alpha = 5.0e-3      # alpha governing viscous evolution
alphaTurb = 5.0e-5  # alpha governing turbulent diffusion
mu = 2.34           # Mean molecular weight
sig_h = 2e-15       # Collisional cross-section of hydrogen atom

# Dust evolution parameters
dustMinSize = 5e-6  # Minimum dust size on dust-grid in dustPy
dustMaxSize = 100   # Maximum dust size on dust-grid in dustPy
allowDriftingParticles = False
                    # Do or don't allow initially drifting particles in dustpy
Rin_dust = 1.0*au   # Inner edge of semimajor axis grid used for dust ev.
Redge_dust = 200*au # Outer edge of semimajor axis grid used for dust ev.
Rnr_dust = 150      # Nr of grid-points on semimajor axis grid used for dust ev.
Rcoarse_int = 50*au # Coarse radial grid spacing in outer disk
Rgrid_out = 1000*au # Outer edge of grid
a_0 = 1.0e-5        # Initial size of dust grains
vfrag = 100.0       # Fragmentation velocity 
rhop = 1.0          # Internal density of dust grains

# Vortex/Planetesimal formation parameters
lifeVortex = 1000   # Lifetime of vortex in units of orbital periods

# Pebble accretion parameters
fracmax = 0.999     # Remove high end of St tail in pebble accretion
lim3d = 10          # If (Racc/(2*Hp))**2 > lim3d, switch to using the
                    # original eq for pebble accretion in the 3d limit
nrOrbInVortex = 10  # Resident time of embryo inside the vortex in 
                    # units of orbital periods
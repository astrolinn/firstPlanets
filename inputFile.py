"""
Contains values of variables and constants
The current values are for an example case
"""
import astropy.constants as c
year = 365.25*24*3600
Msun = c.M_sun.cgs.value

mu = 2.34           # Mean molecular weight
MS = 0.45 * Msun    # Stellar mass
alphaTurb = 0.00001 # Turbulent diffusion parameter
vfrag = 100         # Fragmentation velocity
dt = 500 * year     # Time-step for pebble accretion outside the vortex
tend = 3e6 * year   # Lifetime of protoplanetary disk
tempExp = 0.5       # dlnT/dlnR
sigmaExp = 1.0      # dlnSigma/dlnR at 1au
HExp = 1.25         # dlnH/dlnR
fracmax = 0.999     # Used to prevent numerical issue in pebble accretion routine
lim3d = 10          # Switch to standard 2D pebble accretion
nrOrbInVortex = 10  # Number of orbits the planet accretes pebbles within the vortex

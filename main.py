"""
Main script for running the post-processing calculations
performed in Eriksson et al. (2025) "Planets and 
planetesimals at cosmic dawn: Vortices as planetary nurseries".

This script: 
- Loads existing DustPy data. 
- Calculates the representative Stokes number and maximum 
  dust-to-gas ratios assuming (1) no vortices, (2) vortices 
  that concentrate dust but does not affect dust growth (case VC), 
  and (3) vortices that both concentrate dust and affect dust growth. 
- Calculates where on the space-time grid planetesimals should.
  form based on two different SI criteria
- Calculates mass growth via pebble accretion onto the formed
  embryos.
- Creates some basic plots.

Classes:
    imput_loader: Handles loading the parameters of inputFile.py
    load: Handles loading and trimming the DustPy data
    disk: Handles the calculation of dust-to-gas ratios
    and representative Stokes numbers inside and outside a vortex
    planetesimal: Handles the formation of planetesimals
    pebble: Handles the accretion of pebbles onto a planet
    plot: Contains some simple functions for plotting the results

Usage:
    Run this script directly to execute the full simulation
    inside the simulation directory specified by path.
    NOTE 1: The grid used for the pebble accretion part is currently 
    hadrcoded inside this script.
    NOTE 2: The trimming of the DustPy is currently hardcoded
    inside this script

Example:
    $ python -u main.py --path pathToDir
"""

import argparse
import astropy.constants as c
import numpy as np

import input_loader
import load
import disk
import planetesimal
import pebble
import plot

year = 365.25*24*3600
au = c.au.cgs.value

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to dustPy data.', default='./', type=str)
    args = parser.parse_args()

    # Load parameters from inputFile.py
    input_file = input_loader.load_input_file(args.path)
    Mstar = input_file.Mstar
    nrOrbInVortex = input_file.nrOrbInVortex

    # Load and trim the DustPy 
    model = load.Model(path=args.path, rmax = 50*au, tmin = 1e4*year)

    print('Finished with loading dustpy data')

    # Calculate the representative St and max dust-to-gas ratio
    d2gSt = disk.Disk(model)

    print('Finished with disk module')

    # Planetesimal formation module
    pfEm = planetesimal.Planetesimal(model, d2gSt)
    # Calculations for SI criteria Lim2024b (published version is Lim et al. 2025)
    pf = pfEm.planForm(d2gSt.d2gRho, pfEm.SI_Lim2024b(d2gSt.St_mono))
    pf_vortex = pfEm.planForm(d2gSt.d2gRho_vortex, pfEm.SI_Lim2024b(d2gSt.St_mono))
    pf_vortex_growth = pfEm.planForm(d2gSt.d2gRho_vortex_growth, pfEm.SI_Lim2024b(d2gSt.St_vortex_growth))
    membryo = pfEm.embryoMass(pf)
    membryo_vortex = pfEm.embryoMass_vortex(pf_vortex)
    membryo_vortex_growth = pfEm.embryoMass_vortex(pf_vortex_growth)
    # Calculations for SI criteria Lim2024a (published version is Lim et al. 2024)
    pf_a = pfEm.planForm(d2gSt.d2gRho, pfEm.SI_Lim2024a(d2gSt.St_mono))
    pf_vortex_a = pfEm.planForm(d2gSt.d2gRho_vortex, pfEm.SI_Lim2024a(d2gSt.St_mono))
    pf_vortex_growth_a = pfEm.planForm(d2gSt.d2gRho_vortex_growth, pfEm.SI_Lim2024a(d2gSt.St_vortex_growth))
    membryo_a = pfEm.embryoMass(pf_a)
    membryo_vortex_a = pfEm.embryoMass_vortex(pf_vortex_a)
    membryo_vortex_growth_a = pfEm.embryoMass_vortex(pf_vortex_growth_a)

    # Plotting module
    plotting = plot.Plot(model, d2gSt, path=args.path)
    # Plot disk evolution and planetesimal formation
    plotting.plotDisk(membryo, membryo_vortex, membryo_vortex_growth, membryo_a, membryo_vortex_a, membryo_vortex_growth_a)
    
    # If memory is an issue, these variables can be deleted from memory
    #del d2gSt
    #del pf, pf_vortex, pf_vortex_growth, pf_a, pf_vortex_a, pf_vortex_growth_a

    print('Finished planetesimal formation')

    # Pebble accretion module
    pebb = pebble.Pebble(model, pfEm)

    # Specify the grid used for planet formation via pebble accretion
    tform = np.arange(model.t[0], 2.9e6*year, 1e5*year)
    rform = np.arange(model.r[0], 50*au, 2.5*au)

    # Initialize arrays for storing data
    tform_exact = np.zeros((len(tform)))
    rform_exact = np.zeros((len(rform))) 
    mp_vortex = np.zeros((len(tform), len(rform))) # Case VC, SI criteria b, pebbAcc outside vortex
    mp_vortex_growth = np.zeros((len(tform), len(rform))) # Case VCG, SI criteria b, pebbAcc outside vortex
    mp_vortex_2 = np.zeros((len(tform), len(rform))) # Case VC, SI criteria b, pebbAcc inside vortex
    mp_vortex_growth_2 = np.zeros((len(tform), len(rform))) # Case VCG, SI criteria b, pebbAcc inside vortex
    mp_vortex_a = np.zeros((len(tform), len(rform))) # Case VC, SI criteria a, pebbAcc outside vortex
    mp_vortex_growth_a = np.zeros((len(tform), len(rform))) # Case VCG, SI criteria a, pebbAcc outside vortex
    mp_vortex_2_a = np.zeros((len(tform), len(rform))) # Case VC, SI criteria a, pebbAcc inside vortex
    mp_vortex_growth_2_a = np.zeros((len(tform), len(rform))) # Case VCG, SI criteria a, pebbAcc inside vortex

    for it in range(len(tform)):
        for ir in range(len(rform)):

            tind, rind = pebb.interp_form(tform[it], rform[ir])
            tform_exact[it] = model.t[tind]
            rform_exact[ir] = model.r[rind]
            # Find the embryo mass
            me_v = membryo_vortex[tind, rind]
            me_v_g = membryo_vortex_growth[tind, rind]
            me_v_a = membryo_vortex_a[tind, rind]
            me_v_g_a = membryo_vortex_growth_a[tind, rind]

            # Only perform pebble accretion if the embryo mass is larger than 0
            if me_v > 0 or me_v_g > 0 or me_v_a > 0 or me_v_g_a > 0:
                mform = max(me_v, me_v_g, me_v_a, me_v_g_a)

                # Pebble accretion purely outside the vortex
                mPlanet = pebb.pebbAcc(tform_exact[it], rform_exact[ir], mform)
                if me_v > 0:
                    mp_vortex[it,ir] = mPlanet
                if me_v_g > 0:
                    mp_vortex_growth[it,ir] = mPlanet
                if me_v_a > 0:
                    mp_vortex_a[it,ir] = mPlanet
                if me_v_g_a > 0:
                    mp_vortex_growth_a[it,ir] = mPlanet

                # Pebble accretion initially inside the vortex
                orbitalPeriod = 2 * np.pi * np.sqrt(rform_exact[ir]**3 / (c.G.cgs.value * Mstar))
                tLeaveVortex = tform_exact[it] + nrOrbInVortex * orbitalPeriod
                mPlanet_afterVortex = pebb.pebbAcc_vortex_v2(tform_exact[it], rform_exact[ir], mform, tLeaveVortex)
                mPlanet = pebb.pebbAcc(tLeaveVortex, rform_exact[ir], mPlanet_afterVortex)
                if me_v > 0:
                    mp_vortex_2[it,ir] = mPlanet
                if me_v_g > 0:
                    mp_vortex_growth_2[it,ir] = mPlanet
                if me_v_a > 0:
                    mp_vortex_2_a[it,ir] = mPlanet
                if me_v_g_a > 0:
                    mp_vortex_growth_2_a[it,ir] = mPlanet

        print('Progress pebble accretion:', it/len(tform))

    # Plot the embryo and planetary masses
    plotting.plotPlanet(membryo_vortex, membryo_vortex_growth, tform_exact, rform_exact,
        mp_vortex, mp_vortex_growth, mp_vortex_2, mp_vortex_growth_2,
        mp_vortex_a, mp_vortex_growth_a, mp_vortex_2_a, mp_vortex_growth_2_a)

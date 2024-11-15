"""
Main script for running the post-processing calculations 
for the paper "First planetesimals..."

This script loads dustPy data saved as numpy arrays, 
calculates the dust-to-gas ratios assuming both
vortices and no vortices, calculates the locations
for planetesimal formation, and grows those 
planetesimals into planets via pebble accretion 

Classes:
    load: Handles reading in the dustPy data
    disk: Handles the calculation of dust-to-gas ratios
    and monodisperse Stokes numbers
    planetesimal: Handles the formation of planetesimals
    pebble: Handles the accretion of pebbles onto a planet
    plot: Contains functions for plotting the results
    inputFile.py: Contains variables and constants

Usage:
    Run this file directly to execute the full simulation
    and plot the data
    Choose what SI criteria to use (SI_Lim2024a or 
    SI_Lim2024b) and specify the grid used to grow planets 
    via pebble accretion (tform, rform)

Example:
    $ python main.py
"""

import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse

import load
import disk
import planetesimal
import pebble
import plot

from inputFile import (
    MS, nrOrbInVortex
)

year = 365.25*24*3600
au = c.au.cgs.value

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to dustPy data.', default='./', type=str)
    args = parser.parse_args()

    model = load.Model(path=args.path, rmax = 50*au, tmin = 1e4*year)

    d2gSt = disk.Disk(model)

    pfEm = planetesimal.Planetesimal(model,d2gSt)

    pf = pfEm.planForm(d2gSt.d2gRho, pfEm.SI_Lim2024b(d2gSt.St_mono))
    pf_vortex = pfEm.planForm(d2gSt.d2gRho_vortex, pfEm.SI_Lim2024b(d2gSt.St_mono))
    pf_vortex_growth = pfEm.planForm(d2gSt.d2gRho_vortex_growth, pfEm.SI_Lim2024b(d2gSt.St_vortex_growth))
    membryo = pfEm.embryoMass(pf)
    membryo_vortex = pfEm.embryoMass_vortex(pf_vortex)
    membryo_vortex_growth = pfEm.embryoMass_vortex(pf_vortex_growth)

    pebb = pebble.Pebble(model, pfEm)

    tform = np.arange(model.t[0], 2.9e6*year, 2e5*year)
    rform = np.arange(model.r[0], 50*au, 5*au)

    mp = np.zeros((len(tform),len(rform)))
    mp_vortex = np.zeros((len(tform),len(rform)))
    mp_vortex_growth = np.zeros((len(tform),len(rform)))
    mp_2 = np.zeros((len(tform),len(rform)))
    mp_vortex_2 = np.zeros((len(tform),len(rform)))
    mp_vortex_growth_2 = np.zeros((len(tform),len(rform)))
    tform_exact = np.zeros((len(tform)))
    rform_exact = np.zeros((len(rform)))
    for it in range(len(tform)):
        for ir in range(len(rform)):
            tind, rind = pebb.interp_form(tform[it], rform[ir])
            tform_exact[it] = model.t[tind]
            rform_exact[ir] = model.r[rind]
            me = membryo[tind,rind]
            me_v = membryo_vortex[tind,rind]
            me_v_g = membryo_vortex_growth[tind,rind]
            if me > 0 or me_v > 0 or me_v_g > 0:
                mform = max(me,me_v,me_v_g)
                # Pebble accretion purely outside the vortex
                mPlanet = pebb.pebbAcc(tform[it],rform[ir],mform)
                if me > 0:
                    mp[it,ir] = mPlanet
                if me_v > 0:
                    mp_vortex[it,ir] = mPlanet
                if me_v_g > 0:
                    mp_vortex_growth[it,ir] = mPlanet
                # Pebble accretion initially inside the vortex 
                orbitalPeriod = 2*np.pi * np.sqrt(rform[ir]**3/(c.G.cgs.value*MS))
                tLeaveVortex = tform[it] + nrOrbInVortex * orbitalPeriod
                mPlanet_afterVortex = pebb.pebbAcc_vortex_v2(tform[it],rform[ir],mform,tLeaveVortex)
                mPlanet = pebb.pebbAcc(tLeaveVortex,rform[ir],mPlanet_afterVortex)
                if me > 0:
                    mp_2[it,ir] = mPlanet
                if me_v > 0:
                    mp_vortex_2[it,ir] = mPlanet
                if me_v_g > 0:
                    mp_vortex_growth_2[it,ir] = mPlanet


    plotting = plot.Plot(model, d2gSt, path=args.path)

    plotting.plotDisk(membryo, membryo_vortex, membryo_vortex_growth)

    plotting.plotPlanet(membryo, membryo_vortex, membryo_vortex_growth, tform_exact, rform_exact, mp, mp_vortex, mp_vortex_growth, mp_2, mp_vortex_2, mp_vortex_growth_2)

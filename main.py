import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt


import argparse
import glob

import load
import disk
import planetesimal
import pebble
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-frag_vel', help='description for option1')
    parser.add_argument('-dust2gas', help='description for option2')
    parser.add_argument('-alpha_turb', help='description for option3')
    args = parser.parse_args()

    model = load.Model(args.frag_vel, args.dust2gas, args.alpha_turb, rmax = 50*c.au.cgs.value, tmin = 1e4*365.25*24*3600)

    d2gSt = disk.Disk(model)

    pfEm = planetesimal.Planetesimal(model,d2gSt)

    # Planetesimal formation and Embryo masses for three different limits
    pf = pfEm.planForm(d2gSt.d2gRho, pfEm.SI_Lim2024b(d2gSt.st_mono))
    pf_vortex = pfEm.planForm(d2gSt.d2gRho_vortex, pfEm.SI_Lim2024b(d2gSt.st_mono))
    pf_vortex_growth = pfEm.planForm(d2gSt.d2gRho_vortex_growth, pfEm.SI_Lim2024b(d2gSt.st_vortex_growth))
    membryo = pfEm.embryoMass(pf)
    membryo_vortex = pfEm.embryoMass(pf_vortex)
    membryo_vortex_growth = pfEm.embryoMass(pf_vortex_growth)

    # Grow planets via pebble accretion
    tform = np.arange(model.t[0], 2.9e6*365.25*24*3600, 5e5*365.25*24*3600)
    rform = np.arange(model.r[0], 50*c.au.cgs.value, 10*c.au.cgs.value)
    mp = np.zeros((len(tform),len(rform)))
    mp_vortex = np.zeros((len(tform),len(rform)))
    mp_vortex_growth = np.zeros((len(tform),len(rform)))
    for it in range(len(tform)):
        for ir in range(len(rform)):
            pebb = pebble.Pebble(model,tform[it],rform[ir],membryo,membryo_vortex,membryo_vortex_growth)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:59:13 2025

@author: titouanrenaud
"""

from ctypes import alignment
import os
import numpy as np
import pandas as pd
import plumerise as pr
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import scienceplots
from matplotlib import rc
from utils import getAtmArray, load_input, getIVESPAtable
from matplotlib.lines import Line2D

init_dir = load_input() #Drive data folder

dir_fig = os.getcwd()

# nputs for nevent=130, nn0=5, nT0=6, nmer=11
inputs_dir = f'{init_dir}/MONTE CARLO RUNS/inputs/inputs_nevent=130_nn0=5_nT0=6_nmer=11'

# outputs, have to select matlab or plumerise version later
outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs'

nmer = 11
nn0, nT0 = 5, 6
nalpha, nbeta = 6, 10
nevent = 130

eqn='dry'
outputs_file = os.path.join(outputs_dir, f'outputs_130_nn0=5_nT0=6_nmer=11_nalpha=6_nbeta=10_plumerise_{eqn}_acheve')
outputs_dry = pd.read_csv(outputs_file, low_memory = False)
MER = np.array(outputs_dry['input_mer'])

in0, iT0 = 0,0
MER_best = MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                    + nn0*nT0*nalpha*nbeta * 5
                                    + nT0*nalpha*nbeta * in0
                                    + nalpha*nbeta * iT0 for ievent in range(nevent)]]

MER_low = MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                    + nn0*nT0*nalpha*nbeta * 0
                                    + nT0*nalpha*nbeta * in0
                                    + nalpha*nbeta * iT0 for ievent in range(nevent)]]

MER_high = MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                    + nn0*nT0*nalpha*nbeta * 10
                                    + nT0*nalpha*nbeta * in0
                                    + nalpha*nbeta * iT0 for ievent in range(nevent)]]

# inputs_dir = os.path.join(inputs_dir, f'inputs_nevent={nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}')

inputs = pd.read_csv(inputs_dir)

input_ID = list(inputs['ivespa_ID'])
input_r0 = list(inputs['input_r0'])
input_u0 = list(inputs['input_u0'])
ninputs = len(input_ID)

ivespa = getIVESPAtable() # events index 25, 55, 71

ivespa_index = list(set([list(ivespa['Ivespa_ID']).index(event) for event in input_ID]))
    
eru_h = np.array(ivespa['TopHeight'][ivespa_index])                   # Plume top height [km a.s.l]
eru_ID = np.array(ivespa['Ivespa_ID'][ivespa_index])                  # Event Ivespa ID

eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][ivespa_index])     # Plume top height uncertainty
eru_vh = np.array(ivespa['VentAltitude'][ivespa_index])/1000          # Vent altitude [km a.s.l]
eru_h_avl = eru_h - eru_vh
eru_year = np.array(ivespa['Year'][ivespa_index])

list_dry, list_wet, list_ice = [], [], []

merlist = np.logspace(1, 9, 600)

# %% choice events

events = [  0,   1,   2,   3,  13,  14,  15,  16,  17,  25,  26,  27,  28,
        29,  48,  65,  66,  67,  69,  73,  75,  80,  81, 110, 114, 119,
       124, 125, 126, 127] ## tropical events

L_wind = []
for event in events:
    file_directory = f'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[event]}.csv'

    atmprofile = pd.read_csv(file_directory)
    atm = getAtmArray(atmprofile)
    n_max = np.max(np.where(atm[:,0]<=40000))
    
    L_wind.append((sum(atm[n_max:,4])/len(atm[n_max:,4]) * 3.6, eru_ID[event], event, n_max, atm[:,0][n_max]))

# %% PLOT
ievent = 69  # MER2010_01
n0, T0 = 0.04, 1200

# Atmospheric profile
file_directory = f'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[ievent]}.csv'
atmprofile = pd.read_csv(file_directory)
atm = getAtmArray(atmprofile)

# Plume bulk density at vent height
Rv = 462        # Gas constant of volcanic gas (water) [J/kg/K]
Ra = 287        # Gas constant of dry air [J/kg/K]
rhos = 1200     # Density of solid pyroclasts [kg/m^3]

MetVals, z_pos = pr.Met_interp(atm, eru_vh[ievent]*1000, z_pos=0)
Pa0 = MetVals["Pa"]
rho_B0 = 1/(n0*Rv*T0/Pa0 + (1-n0)/rhos)

Cs = 1100
# Cs = 1617
Ca = 1000
Cv = 1810
Ra = 287

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

with plt.style.context(["science"]): 
    plt.figure(figsize=(12,7), layout='constrained')
    plt.xscale('log')
    plt.xlim(9.2, 1.52e9)
    plt.ylim(-0.7, 41.3)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.grid(alpha=0.8)
    
    plt.xlabel(r'Mass Eruption Rate $(\overline{MER})\;[kg.s^{-1}]$', fontsize=16)
    plt.ylabel(r'Plume top height ($\overline{H_{top}}$) [km \;a.v.l.]', fontsize=16)
    
    ### IVESPA
    markers, caps, bars = plt.errorbar(MER_best, np.array(eru_h_avl, dtype=float), xerr=[MER_low, MER_high], yerr=eru_d_h/2, marker='o', linestyle='None', ecolor='k', elinewidth=0.5,
                 capsize=1.2, markerfacecolor='k', markersize=3, alpha=0.3, label='IVESPA (with estimated 95\% uncertainty)')
    
    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.1) for cap in caps]

    handles = [markers, caps[0], bars[0]]
    legend1 = plt.legend(fontsize=14, frameon=True, fancybox=True, 
                         framealpha=0.5, loc=[0.02, 0.93], alignment='left')
    
    ### Engwell et al 2023
    d0, = plt.plot(merlist, 0.345*merlist**0.226, lw=1.7, color='k', ls='--')
    
    label=r'Empirical power-law, Engwell et al. 2023 ($\overline{H_{top}}=0.345\times\overline{MER}^{\; 0.226}$)'
    legend2 = plt.legend([d0], [label], title='0D model', title_fontsize=15,
                         fontsize=14, frameon=True, fancybox=True, framealpha=0.5, 
                         loc=[0.02, 0.81], alignment='left')

    ### dry plumerise, a,b=0.1, 0.5
    alpha, beta = 0.1, 0.5
    l1 = []
    for mer in merlist:
        u0 = max((np.log(mer) - 1.08)/0.0234, 7)
        r0 = np.sqrt(mer/( np.pi * u0 * rho_B0 ))
        
        eqn = 'dry'
        Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                               Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

        SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

        IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=eru_vh[ievent]*1000)
        
        soln, plume_state, flag = pr.solve(Params, SolParams, IC, atm, verbose=False)
            
        z = pr.get_soln(soln,'z')          #altitude [m]
        
        z_max = max(z)
        z_max = z_max/1000 - eru_vh[ievent]
        l1.append(z_max)
        
    n_max = np.max(np.where(atm[:,0]<=40000))

    l5, = plt.plot(merlist, l1, lw=1.7, ls='-', color='#cd1866')
    labels = [r'dry Plumerise, $(\alpha, \beta)=(0.1, 0.5)$']

    ### dry plumerise, a,b=0.17, 1.0
    alpha, beta = 0.17, 1.0
    l1 = []
    for mer in merlist:
        u0 = max((np.log(mer) - 1.08)/0.0234, 7)
        r0 = np.sqrt(mer/( np.pi * u0 * rho_B0 ))
        
        eqn = 'dry'
        Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                               Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

        SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

        IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=eru_vh[ievent]*1000)
        
        soln,plume_state,flag = pr.solve(Params, SolParams, IC, atm, verbose=False)
            
        z = pr.get_soln(soln,'z')          #altitude [m]
        
        z_max = max(z)
        z_max = z_max/1000 - eru_vh[ievent]
        l1.append(z_max)
        
    l2, = plt.plot(merlist, l1, lw=1.7, ls='-', color="#812913")
    labels.append(r'dry Plumerise, $(\alpha, \beta)=(0.17, 1.0)$')
    
    ### dry plumerise, a,b=0.05, 0.1
    alpha, beta = 0.05, 0.1
    l1 = []
    for mer in merlist:
        u0 = max((np.log(mer) - 1.08)/0.0234, 7)
        r0 = np.sqrt(mer/( np.pi * u0 * rho_B0 ))
        
        eqn = 'dry'
        Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                               Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

        SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

        IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=eru_vh[ievent]*1000)
        
        soln,plume_state,flag = pr.solve(Params, SolParams, IC, atm, verbose=False)
            
        z = pr.get_soln(soln,'z')          #altitude [m]
        
        z_max = max(z)
        z_max = z_max/1000 - eru_vh[ievent]
        l1.append(z_max)
        
    l3, = plt.plot(merlist, l1, lw=1.7, ls='-', color="#f9b659")
    labels.append(r'dry Plumerise, $(\alpha, \beta)=(0.05, 0.1)$')
    
    ### ice plumerise, a,b=0.1, 0.5
    alpha, beta = 0.1, 0.5
    l1 = []
    for mer in merlist:
        u0 = max((np.log(mer) - 1.08)/0.0234, 7)
        r0 = np.sqrt(mer/( np.pi * u0 * rho_B0 ))
        
        eqn = 'ice'
        Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                               Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

        SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

        IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=eru_vh[ievent]*1000)
        
        soln,plume_state,flag = pr.solve(Params, SolParams, IC, atm, verbose=False)
            
        z = pr.get_soln(soln,'z')          #altitude [m]
        
        z_max = max(z)
        z_max = z_max/1000 - eru_vh[ievent]
        l1.append(z_max)
        
    l4, = plt.plot(merlist, l1, lw=1.7, ls='-.', color="#cd1866")
    labels.append(r'ice Plumerise, $(\alpha, \beta)=(0.1, 0.5)$')
    
    legend3 = plt.legend([l5, l2, l3, l4], labels, title='1D models', fontsize=14, 
                         title_fontsize=15, frameon=True, fancybox=True, framealpha=0.5, 
                         loc=[0.02, 0.565], alignment='left')

    ### 3D model Placeholders at the time

    mer3D = np.array([1e3, 1e5, 1e7])
    h3D = np.array([2.5, 10, 25]) 
    d3, = plt.plot(mer3D, h3D, marker='s', color="#196e32", linestyle='None', markersize=5, label='3D model placeholders')

    custom_handle = Line2D(
    [], [], 
    marker='s',
    linestyle='',
    color='#196e32',
    markersize=5)

    plt.legend(handles=[custom_handle], labels=[r'3D model placeholders'], title='3D model', title_fontsize=15, 
                         fontsize=14, frameon=True, fancybox=True, framealpha=0.5, 
                         loc=[0.02, 0.46], alignment='left')
    
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)

    # plt.savefig(os.path.join(dir_fig, 'Figure1.pdf'), dpi=300, format='pdf')
    # plt.close()
    plt.show()
   
# %%

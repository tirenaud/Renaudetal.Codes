#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:50:24 2025

@author: titouanrenaud
"""

import os
import numpy as np
import pandas as pd
import plumerise as pr
import random
import time
import matplotlib.pyplot as plt
from utils import getAtmArray, getIVESPAtable, load_input

init_dir = load_input()

ivespa = getIVESPAtable() # events index 25, 55, 71

###  Extract IVESPA's data for all IVESPA events ###

eru_h = np.array(ivespa['TopHeight'])                   # Plume top height [km a.s.l]

list_indexes = []

for i in range(len(eru_h)):
    if eru_h[i] != 'Unknown':
        list_indexes.append(i)
        
nevent = len(list_indexes)
    
eru_h = np.array(ivespa['TopHeight'][list_indexes])                   # Plume top height [km a.s.l]
eru_ID = np.array(ivespa['Ivespa_ID'][list_indexes])                  # Event Ivespa ID

eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][list_indexes])     # Plume top height uncertainty
eru_dur = np.array(ivespa['Duration'][list_indexes]*3600)             # Duration of the event [s]
eru_d_dur = np.array(ivespa['Duration_Uncertainty'][list_indexes])*3600
eru_vh = np.array(ivespa['VentAltitude'][list_indexes])/1000          # Vent altitude [km a.s.l]
eru_tem = np.array(ivespa['TEM'][list_indexes])                       # Total Erupted Mass [kg]
eru_d_tem_low = np.array(ivespa['TEM_UncertaintyLow'][list_indexes])
eru_d_tem_high = np.array(ivespa['TEM_UncertaintyHigh'][list_indexes])

for ieru in range(nevent):
    if eru_d_tem_low[ieru] == 'Unknown':
        eru_d_tem_low[ieru] = 2/3*eru_tem[ieru]
    if eru_d_tem_high[ieru] == 'Unknown':
        eru_d_tem_high[ieru] = 2*eru_tem[ieru]
        
# %% Create arrays for S1

T0 = 900

list_u0_min = np.array([[5.0]*130]*5)

for ievent in range(130):
    vh0 = eru_vh[ievent]*1000
    
    # Atmospheric profile
    file_directory = f'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[ievent]}.csv'
    
    atmprofile = pd.read_csv(file_directory)
    atm = getAtmArray(atmprofile)
    
    # Plume bulk density at vent height
    Rv = 462        # Gas constant of volcanic gas (water) [J/kg/K]
    Ra = 287        # Gas constant of dry air [J/kg/K]
    rhos = 1200     # Density of solid pyroclasts [kg/m^3]
    
    for in0 in range(5):
        n0 = np.linspace(0.01, 0.07, num=5)[in0]
    
        MetVals, z_pos = pr.Met_interp(atm, eru_vh[ieru], z_pos=0)
        Pa0 = MetVals["Pa"]
        rho_B0 = 1/(n0*Rv*T0/Pa0 + (1-n0)/rhos)
        
        # mer_low = eru_d_tem_low[ievent] / ( (eru_dur[ievent]+eru_d_dur[ievent]) * (1-n0))
        mer_high = eru_d_tem_high[ievent] / ( (eru_dur[ievent]-eru_d_dur[ievent]) * (1-n0))
        
        list_u0 = np.linspace(5, 600, num =200)
        list_r0 = []
        list_Htop = []
        
        Cs = 1100
        Ca = 1000
        Cv = 1810
        Ra = 287
        
        alpha, beta = 0.05, 0.05
        eqn = 'dry'
        
        for u0 in list_u0:
            r0 = np.sqrt(mer_high/( np.pi * u0 * rho_B0 ))  
            list_r0.append(r0)
                
            Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                                   Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)
            
            SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)
            
            IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=vh0)
            
            soln,plume_state,flag= pr.solve(Params, SolParams, IC, atm, verbose=False)
            
            if plume_state['Collapse']:
                list_u0_min[in0, ievent] = u0
                
            else:
                break
                
    print(eru_ID[ievent])

# %% Create arrays for S2

T0 = 900
n0 = 0.01

list_u0_min = np.array([5.0]*130)
list_best_mer = []
Htop_results = np.zeros((130, 200))

list_u0 = np.linspace(5, 600, num =200)

for ievent in range(130):
    vh0 = eru_vh[ievent]*1000
    
    # Atmospheric profile
    file_directory = f'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[ievent]}.csv'
    
    atmprofile = pd.read_csv(file_directory)
    atm = getAtmArray(atmprofile)
    
    # Plume bulk density at vent height
    Rv = 462        # Gas constant of volcanic gas (water) [J/kg/K]
    Ra = 287        # Gas constant of dry air [J/kg/K]
    rhos = 1200     # Density of solid pyroclasts [kg/m^3]
    
    best_mer = eru_tem[ievent] / ( eru_dur[ievent] * (1-0.04) )
    list_best_mer.append(best_mer)
    mer_high = eru_d_tem_high[ievent] / ( (eru_dur[ievent]-eru_d_dur[ievent]) * (1-n0))

    MetVals, z_pos = pr.Met_interp(atm, eru_vh[ieru], z_pos=0)
    Pa0 = MetVals["Pa"]
    rho_B0 = 1/(n0*Rv*T0/Pa0 + (1-n0)/rhos)

    list_r0 = []
    list_Htop = []
    
    Cs = 1100
    Ca = 1000
    Cv = 1810
    Ra = 287
    
    alpha, beta = 0.05, 0.05
    eqn = 'dry'
    
    k=0
    for u0 in list_u0:
        # r0 = np.sqrt(mer_high/( np.pi * u0 * rho_B0 ))  
        r0 = np.sqrt(best_mer/( np.pi * u0 * rho_B0 ))  
        list_r0.append(r0)
            
        Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                               Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)
        
        SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)
        
        IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=vh0)
        
        soln,plume_state,flag = pr.solve(Params, SolParams, IC, atm, verbose=False)
        
        if plume_state['Collapse']:
            Htop_results[ievent, k] = 0
            
        else:
            Htop_results[ievent, k] = plume_state['Htop']

        k += 1

    print(eru_ID[ievent])

# %% FIGURE S1 MER VS u0 min to avoid collapse

plt.figure(figsize=(10, 6))

plt.ylabel('Mass Eruption Rate [kg.s$^{-1}$]', fontsize=15)
plt.xlabel('Minimal $u_0$ avoiding plume collapse [m.s$^{-1}$]', fontsize=15)

plt.yscale('log')

plt.grid(alpha=0.6)

plt.plot(list_u0_min, list_best_mer, 'o')

plt.plot(list_u0_min, 10**(1.08+0.0234*list_u0_min), color='green')
plt.text(180, 8e4, 'log(MER) = 1.08+0.0234 u0', fontsize=14, color='green')

for ievent in range(130):
    plt.text(list_u0_min[ievent], list_best_mer[ievent]*1.3, eru_ID[ievent], fontsize=5, rotation=90, horizontalalignment='center', verticalalignment='bottom')

plt.savefig('FigureS1_u_0_min-collapse.pdf', bbox_inches='tight', dpi=300, format='pdf')

# %% FIGURE S2 Htop VS u0 with 0 height when collapse

plt.figure(figsize=(10, 6))

plt.ylabel('Predicted top height [km a.v.l.]', fontsize=15)
plt.xlabel('Initial vertical velocity $u_0$ [m.s$^{-1}$]', fontsize=15)

plt.grid(alpha=0.6)

for ievent in range(130):
    plt.plot(list_u0, Htop_results[ievent,:]/1000, ls='--', lw=0.7)

plt.savefig('FigureS2_Htop_VS_u0_collapse.pdf', bbox_inches='tight', dpi=300, format='pdf')

# %%

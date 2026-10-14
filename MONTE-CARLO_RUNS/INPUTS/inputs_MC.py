#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:21:57 2025

@author: titouanrenaud

This script defines the inputs used for running the models:
    * u0 initial plume vertical velocity (m/s)
    * r0 vent radius / initial radius of plume (m)
    * n0 initial gas mas fraction (1)
    * T0 initial plume temperature (K)
"""

import os
import plumerise as pr
import numpy as np
import pandas as pd
from utils import getAtmArray, load_input, getIVESPAtable

init_dir = load_input() #Drive data folder

inputs_dir = os.getcwd()
ivespa = getIVESPAtable()

###  Extract IVESPA's data for all IVESPA events ###

eru_h = np.array(ivespa['TopHeight'])                                 # Plume top height [km a.s.l]
list_indexes = np.nonzero(eru_h != 'Unknown')[0] # keep the 130 events with known top heights
nevent = len(list_indexes)
    
eru_ID = np.array(ivespa['Ivespa_ID'][list_indexes])                  # Event Ivespa ID

eru_h = np.array(ivespa['TopHeight'][list_indexes])                   # Plume top height [km a.s.l]
eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][list_indexes])     # Plume top height uncertainty

eru_dur = np.array(ivespa['Duration'][list_indexes]*3600)             # Duration of the event [s]
eru_d_dur = np.array(ivespa['Duration_Uncertainty'][list_indexes])*3600

eru_vh = np.array(ivespa['VentAltitude'][list_indexes])/1000          # Vent altitude [km a.s.l]

eru_tem = np.array(ivespa['TEM'][list_indexes])                       # Total Erupted Mass [kg]
eru_d_tem_low = np.array(ivespa['TEM_UncertaintyLow'][list_indexes])
eru_d_tem_high = np.array(ivespa['TEM_UncertaintyHigh'][list_indexes])

for ieru in range(nevent): # adding uncertainties on TEM when unknown
    if eru_d_tem_low[ieru] == 'Unknown':
        eru_d_tem_low[ieru] = 2/3*eru_tem[ieru]
    if eru_d_tem_high[ieru] == 'Unknown':
        eru_d_tem_high[ieru] = 2*eru_tem[ieru]
 
# Resampling n0, T0 & MER
nn0, nT0 = 5, 6
n0list = np.linspace(0.01, 0.07, num=nn0)
n0list = [round(n0list[i], 15) for i in range(nn0)]
T0list = np.linspace(900, 1500, num=nT0)
nmer = 11

nresamp = nn0*nT0*nmer

### Initialize arrays to save the input parameters ###
# Size = number of events x number of sets of inputs (n0, T0, MER)

input_r0 = np.zeros((nevent,nresamp))                       # Vent radius [m]
input_u0 = np.zeros((nevent,nresamp))                       # Exit velocity [m/s]
input_rhoB0 = np.zeros((nevent,nresamp))                    # Plume bulk density at vent [kg/m3]
input_n0 = np.array([np.array([n0list[in0] for k in range(nmer) for in0 in range(len(n0list)) for iT0 in range(len(T0list))])
            for ievent in range(nevent)])
input_T0 = np.array([np.array([T0list[iT0] for k in range(nmer) for in0 in range(len(n0list)) for iT0 in range(len(T0list))])
            for ievent in range(nevent)])
input_mer = np.zeros((nevent,nresamp))                      # Resampled mass eruption rate [kg/s]

#%% Loop for drawing
 
for ievent in range(nevent):
    # Resampling the MER, nmer values bewteen low and high estimate
    best_mer = eru_tem[ievent] / ( eru_dur[ievent] * (1-0.04) )
    mer_low = (eru_tem[ievent]-eru_d_tem_low[ievent]) / ( (eru_dur[ievent]+eru_d_dur[ievent]) * (1-n0list[0]))
    mer_high = (eru_tem[ievent]+eru_d_tem_high[ievent]) / ( (eru_dur[ievent]-eru_d_dur[ievent]) * (1-n0list[-1]))
       
    list_mer = []
    list_low = np.linspace(mer_low, best_mer, nmer//2+1)
    for i in range(len(list_low)):
        list_mer.append(list_low[i])
    list_high = np.linspace(best_mer, mer_high, nmer//2+1)
    for i in range(1, len(list_high)):
        list_mer.append(list_high[i])
    
    input_mer[ievent, :] = [list_mer[k] for k in range(nmer) for in0 in range(len(n0list)) for iT0 in range(len(T0list)) ]
    
    for k in range(nmer):
        mer = list_mer[k]
        for in0 in range(nn0):
            n0 = n0list[in0]
            for iT0 in range(nT0):
                T0 = T0list[iT0]
                
                # Atmospheric profile
                file_directory = rf'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[ievent]}.csv'
    
                atmprofile = pd.read_csv(file_directory)
                atm = getAtmArray(atmprofile)
                
                # Plume bulk density at vent height
                Rv = 462        # Gas constant of volcanic gas (water) [J/kg/K]
                Ra = 287        # Gas constant of dry air [J/kg/K]
                rhos = 1200     # Density of solid pyroclasts [kg/m^3]
                
                MetVals, z_pos = pr.Met_interp(atm, eru_vh[ievent]*1000, z_pos=0) # atmospheric conditions at vent
                Pa0 = MetVals["Pa"] # atmospheric pressure at vent
                rho_B0 = 1/(n0*Rv*T0/Pa0 + (1-n0)/rhos) # calculation of initial bulk density
                
                r0, u0, success = pr.draw_u0_r0_gauss(mer, rho_B0, eru_vh[ievent]*1000, T0, n0, atm)
                
                input_r0[ievent, nn0*nT0*k + nT0*in0 + iT0] = r0
                input_u0[ievent, nn0*nT0*k + nT0*in0 + iT0] = u0
                input_rhoB0[ievent, nn0*nT0*k + nT0*in0 + iT0] = rho_B0
                
                if not success:
                    print(f'fail for event {eru_ID[ievent]}, k={k}, in0={in0}, iT0={iT0}')
                    
    print(ievent)

frame_inputs = {"ivespa_ID":[eru_ID[i]for i in range(nevent) for k in range(nresamp)], 
                "input_r0":[input_r0[i,k] for i in range(nevent) for k in range(nresamp)], 
                "input_u0":[input_u0[i,k] for i in range(nevent) for k in range(nresamp)],
                "input_n0":[input_n0[i,k] for i in range(nevent) for k in range(nresamp)], 
                "input_T0":[input_T0[i,k] for i in range(nevent) for k in range(nresamp)],
                "input_mer":[input_mer[i,k] for i in range(nevent) for k in range(nresamp)],
                "input_rhoB0":[input_rhoB0[i,k] for i in range(nevent) for k in range(nresamp)]}
    
inputs = pd.DataFrame(frame_inputs)

inputs_file = os.path.join(inputs_dir, f'inputs_nevent={nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}')

inputs.to_csv(inputs_file, index=False)

        
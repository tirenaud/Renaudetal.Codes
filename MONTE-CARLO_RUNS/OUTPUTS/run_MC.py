#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:24:04 2025

@author: titouanrenaud
"""

import os
from sre_constants import IN
import numpy as np
import pandas as pd
import plumerise as pr
from joblib import Parallel, delayed
import time
from pathlib import Path
from utils import getAtmArray, load_input, getIVESPAtable

init_dir = load_input() #Drive data folder

inputs_dir = f'{init_dir}/MONTE CARLO RUNS/inputs/' #where the inputs file is located

nevent = 130
nmer = 11
nn0, nT0 = 5, 6

inputs_file = os.path.join(inputs_dir, f'inputs_nevent={nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}')

inputs = pd.read_csv(inputs_file)

input_ID = list(inputs['ivespa_ID'])
input_r0 = list(inputs['input_r0'])
input_u0 = list(inputs['input_u0'])
input_n0 = list(inputs['input_n0'])
input_T0 = list(inputs['input_T0'])
input_mer = list(inputs['input_mer'])

ninputs = len(input_ID)

ivespa = getIVESPAtable()

ivespa_index = list(set([list(ivespa['Ivespa_ID']).index(event) for event in input_ID]))
    
eru_ID = np.array(ivespa['Ivespa_ID'][ivespa_index])                  # Event Ivespa ID

eru_h = np.array(ivespa['TopHeight'][ivespa_index])                   # Plume top height [km a.s.l]
eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][ivespa_index])     # Plume top height uncertainty

eru_dur = np.array(ivespa['Duration'][ivespa_index]*3600)             # Duration of the event [s]
eru_d_dur = np.array(ivespa['Duration_Uncertainty'][ivespa_index])*3600

eru_vh = np.array(ivespa['VentAltitude'][ivespa_index])/1000          # Vent altitude [km a.s.l]

eru_tem = np.array(ivespa['TEM'][ivespa_index])                       # Total Erupted Mass [kg]
eru_d_tem_low = np.array(ivespa['TEM_UncertaintyLow'][ivespa_index])
eru_d_tem_high = np.array(ivespa['TEM_UncertaintyHigh'][ivespa_index])

for ieru in range(nevent):
    if eru_d_tem_low[ieru] == 'Unknown':
        eru_d_tem_low[ieru] = 2/3*eru_tem[ieru]
    if eru_d_tem_high[ieru] == 'Unknown':
        eru_d_tem_high[ieru] = 2*eru_tem[ieru]

# Tested entrainment parametrizations

alphalist = np.linspace(0.05, 0.2, num=6)
betalist = np.linspace(0.05, 0.95, num=10)

nalpha, nbeta = len(alphalist), len(betalist)

outputs_dir = os.getcwd()

#%% Loop for events
  
eqn = 'dry'

temps0=time.time()
temps=time.time()

# being able to stop the iterations at any point. Will end up compiling the current batch and will stop there
stop_requested=[False]

try:
    low_memory = False
    outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_inacheve')
    outputs = pd.read_csv(outputs_file)
    
    print('Restarting MC where it stopped')
    
    i_input_start = int(max(outputs["i_input"])+2)
    
    # os.remove(outputs_file)
    
except:
    list_IDs = np.empty(ninputs*nalpha*nbeta)
    list_IDs[:] = np.nan
    list_IDs = pd.Series(list_IDs, dtype='string')

    frame_outputs = {"ivespa_ID": np.zeros(ninputs*nalpha*nbeta, dtype='str'),
                     "input_mer": np.zeros(ninputs*nalpha*nbeta),
                     "input_n0": np.zeros(ninputs*nalpha*nbeta),
                     "input_T0": np.zeros(ninputs*nalpha*nbeta),
                     "input_r0": np.zeros(ninputs*nalpha*nbeta),
                     "input_u0": np.zeros(ninputs*nalpha*nbeta),
                     "i_input": np.zeros(ninputs*nalpha*nbeta),
                     "alpha": np.zeros(ninputs*nalpha*nbeta),
                     "beta": np.zeros(ninputs*nalpha*nbeta),
                     "Htop": np.zeros(ninputs*nalpha*nbeta), # [km a.v.l]
                     "Ltop": np.zeros(ninputs*nalpha*nbeta), # [m]
                     "Hb": np.zeros(ninputs*nalpha*nbeta),
                     "output_mer": np.zeros(ninputs*nalpha*nbeta),
                     "Collapse": np.zeros(ninputs*nalpha*nbeta),
                     "SolverFlag": np.zeros(ninputs*nalpha*nbeta)}
    
    outputs = pd.DataFrame(frame_outputs)
    
    i_input_start = 0    


Cs = 1100
# Cs = 1617
Ca = 1000
Cv = 1810
Ra = 287

#iterable function (what's running in the loops)
def process_combination(i_input, ialpha, ibeta):
    
    r0 = input_r0[i_input]
    u0 = input_u0[i_input]  
    n0 = input_n0[i_input]
    T0 = input_T0[i_input]
    
    i_eru = i_input // (nn0*nT0*nmer)
    vh0 = eru_vh[i_eru]*1000
    
    #we load the eruption atmospheric profile
    file_directory = rf'{init_dir}/ATMOSPHERIC PROFILES/ERA_reanalysis_profiles/time_averaged/atm_ERAreanalysis_{eru_ID[i_eru]}.csv'

    atmprofile = pd.read_csv(file_directory)
    atm = getAtmArray(atmprofile)

    #set up of the model input params
    alpha = alphalist[ialpha]
    beta = betalist[ibeta]
    
    Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                           Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

    SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

    IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=vh0)
       
    # we try to solve unless we get the unfamous division by zero error (this should no longer be relevant)
    t1 = time.time()
    while time.time()-t1 < 30:
        try:
            soln,plume_state,flag, MetVals = pr.solve(Params, SolParams, IC, atm, verbose=False)
                
            z = pr.get_soln(soln,'z')               #altitude [m]
            L = pr.get_soln(soln,'L')               #plume radius [m]
            rho_0 = pr.get_soln(soln, 'rho')[0]
            collapse = plume_state['Collapse']      #collapse flag
            neutr = plume_state['Hn']               #neutral buoyancy height [m]
            
            z_max = max(z)
            ind = list(z).index(z_max)
            z_max = z_max/1000 - eru_vh[i_eru]
            
            mer = np.pi * rho_0 * r0**2 * u0
    
            soln_n = [input_ID[i_input], input_mer[i_input], input_n0[i_input], input_T0[i_input], input_r0[i_input],
                      input_u0[i_input], i_input, alpha, beta, z_max, L[ind], neutr, mer, collapse, flag]
    
            return soln_n
                
        except :
            return f"error for coordinates i_input={i_input}, {eru_ID[i_eru]}, {alpha}, {beta}"
        
    return [input_ID[i_input], input_mer[i_input], input_n0[i_input], input_T0[i_input], input_r0[i_input],
              input_u0[i_input], i_input, alpha, beta, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

for i_input in range(i_input_start, ninputs):
    if stop_requested[0]:
        break
        
    print(f'input =  {i_input} / {ninputs}, time for last MC = {round(time.time()-temps,2)} s')
    temps = time.time()
    
    if i_input>0 and i_input % 100 == 0:
        temps_estime_restant = (ninputs - i_input)/(i_input - i_input_start) * round(temps-temps0, 2)
        heures = temps_estime_restant // 3600
        minutes = (temps_estime_restant - heures*3600)//60
        print(f'temps restant estimé: {heures} heures et {minutes} minutes')
    
    outputs_batch = Parallel(n_jobs=-1)(
        delayed(process_combination)(i_input, ialpha, ibeta)
        for ialpha in range(nalpha)
        for ibeta in range(nbeta)
    )
    
    outputs.loc[[nalpha*nbeta * i_input + nbeta * ialpha + ibeta
                 for ialpha in range(nalpha)
                 for ibeta in range(nbeta)], :] = outputs_batch
   
if i_input == ninputs-1: # if all batches finished
    outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')

else : # if stoped mid-calculation
    outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_inacheve')

outputs.to_csv(outputs_file, index=False)

print(f"MC done, files saved as {outputs_file}")

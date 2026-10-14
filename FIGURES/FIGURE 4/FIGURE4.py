#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:36:07 2025

@author: titouanrenaud
"""
# %% Imports

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
import time
from utils import load_input

init_dir = load_input() #Drive folder

dir_fig = os.getcwd() #directory to save figures

outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs' #where to find models' output

nevent = 130
nn0, nT0 = 5, 6
nmer = 11
nalpha, nbeta = 6, 10

eqn='dry' #uploading dry Plumerise's outputs
outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs_dry = pd.read_csv(outputs_file, low_memory = False)
Htop_dry = np.array(outputs_dry['Htop'])
MER = np.array(outputs_dry['input_mer'])

eqn = 'wet' #uploading wet Plumerise's outputs
outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs = pd.read_csv(outputs_file, low_memory = False)
Htop_wet = np.array(outputs['Htop'])

eqn = 'ice' #uploading ice Plumerise's outputs
outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs = pd.read_csv(outputs_file, low_memory = False)
Htop_ice = np.array(outputs['Htop'])

n0list = list(set(list(outputs_dry['input_n0'])))
n0list.sort()

T0list = list(set(list(outputs_dry['input_T0'])))
T0list.sort()

alphalist = list(set(list(outputs_dry['alpha'])))
alphalist.sort()

betalist = list(set(list(outputs_dry['beta'])))
betalist.sort()

ivespa = pr.getIVESPAtable() #uploading IVESPA table

ivespa_ID = list(outputs.ivespa_ID[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                for ievent in range(nevent)]])

ivespa_indexes = [list(ivespa['Ivespa_ID']).index(event) for event in ivespa_ID]

eru_h = np.array(ivespa['TopHeight'][ivespa_indexes])                   # Plume top height [km a.s.l]
eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][ivespa_indexes])     # Plume top height uncertainty
eru_vh = np.array(ivespa['VentAltitude'][ivespa_indexes])/1000          # Vent altitude [m a.s.l]
eru_h_avl = eru_h - eru_vh
eru_dur = np.array(ivespa['Duration'][ivespa_indexes])

save_directory = f'{init_dir}/MONTE CARLO RUNS/outputs/MATLAB/L=0.0/' #uploading Geneva lambda=0.0 outputs
out_Htop0 = np.array(pd.read_csv(save_directory+'out_Htop.csv', low_memory = False, header=None))

save_directory = f'{init_dir}/MONTE CARLO RUNS/outputs/MATLAB/L=1e-2/' #uploading Geneva lambda=1e-2 outputs
out_Htop2 = np.array(pd.read_csv(save_directory+'out_Htop.csv', low_memory = False, header=None))

# %% DELTA H(ievent) ice
"""

Calculating DELTA_H. A wheight accounting for uncertainties over observed top height and for propagation of uncertainties
over MER and (n0, T0) threw the models.
Taking only DELTA_H for ice_plumerise to have same wheight for all the models and having results allowing comparison 
between the models. Ice plumerise has the widest range of outputs, this is why we take this model to calculate DELTA_H.

"""

nmc = 10000 #10 000 Monte Carlo simulations. 
        
### resampling the observed top height. Gaussian centered on the best-estimate and boudaries uncertainties given in IVESPA
rand_h2 = np.zeros((nevent, nmc))
for ievent in range(nevent):
    for i in range(nmc):
        while True:
            rand_h2[ievent, i] = random.gauss(eru_h_avl[ievent], eru_d_h[ievent]/2)
            if abs(rand_h2[ievent, i]-eru_h_avl[ievent]) < 2*eru_d_h[ievent] and rand_h2[ievent, i] > 0: # get rid of extreme values
                break

### resampling the models' outputs. Chossing indexes in the discreet Monte Carlo for n0, T0 and MER
n0_indexes = []
T0_indexes = []
mer_indexes = []

for imc in range(nmc):
    in0 = random.randint(0, nn0-1)
    n0_indexes.append(in0)
    
    iT0 = random.randint(0, nT0-1)
    T0_indexes.append(iT0)
    
    imer = int(round(random.gauss(nmer//2, nmer/4), 0))
    while imer not in range(nmer):
        imer = int(round(random.gauss(nmer//2, nmer/4), 0))
    mer_indexes.append(imer)
    

def func1(ievent): #function to calculate sigma^2, giving next the value for DELTA_H 
    mu1 = np.array([ sum(rand_h2[ievent,:] - Htop_ice[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                           + nn0*nT0*nalpha*nbeta * mer_indexes[imc]
                                           + nT0*nalpha*nbeta * n0_indexes[imc]
                                           + nalpha*nbeta * T0_indexes[imc]
                                           + nbeta*ialpha + ibeta for imc in range(nmc)]])/nmc for ialpha in range(nalpha) for ibeta in range(nbeta)] )
            
    aux1 = np.array([ sum((rand_h2[ievent,:] - Htop_ice[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                           + nn0*nT0*nalpha*nbeta * mer_indexes[imc]
                                           + nT0*nalpha*nbeta * n0_indexes[imc]
                                           + nalpha*nbeta * T0_indexes[imc]
                                           + nbeta*ialpha + ibeta for imc in range(nmc)]] - mu1[nbeta*ialpha+ibeta])**2 ) for ialpha in range(nalpha) for ibeta in range(nbeta) ])
            
    sigma1 = 1/(nmc*nalpha*nbeta) * np.sum(aux1[:])
    return [ivespa_ID[ievent], sigma1]
    
out = Parallel(n_jobs=-1)(
    delayed(func1)(ievent)
    for ievent in range(nevent)
)

out=np.array(out)

delta_h = 1.97*np.sqrt(np.array(out[:,1], dtype='float64')) #DELTA_H for ice Plumerise
delta_norm = delta_h / eru_h_avl #Normalized DELTA_H, divided by observed top height


# %% ALL EVENTS

entr_all = np.zeros((5, nmc, 2))
better_0D_all = np.zeros((5, nmc))
bett_0D_all = np.zeros(5)

events = np.array(range(nevent)) #defining the list of events taken. Here all 130 events, no selection.
ngood = nevent #number of events in the selection. Here 130

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_all[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_all[i, imc] = len(np.nonzero(better)[0]) / ngood
        bett_0D_all[i] = (len(np.nonzero(better)[0])>len(events)/2)
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_all[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_all[i, imc] = len(np.nonzero(better)[0]) / ngood
        bett_0D_all[i] = (len(np.nonzero(better)[0])>len(events)/2)
        
    i += 1
    
    
# %% OUTLIERS

entr_outliers = np.zeros((5, nmc, 2))
better_0D_outliers = np.zeros((5, nmc))

list_under, list_over = [], []

for ievent in range(nevent):
    lim_under, lim_over = eru_h_avl[ievent]-eru_d_h[ievent]/2, eru_h_avl[ievent]+eru_d_h[ievent]/2
    Lout =np.concatenate([ Htop_dry[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                           + nn0*nT0*nalpha*nbeta * imer
                                           + nT0*nalpha*nbeta * in0
                                           + nalpha*nbeta * iT0
                                           + nbeta*ialpha + ibeta for imer in range(nmer) for in0 in range(nn0) for iT0 in range(nT0) for ialpha in range(4) for ibeta in range(1,10)]],
                          Htop_wet[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                            + nn0*nT0*nalpha*nbeta * imer
                                            + nT0*nalpha*nbeta * in0
                                            + nalpha*nbeta * iT0
                                            + nbeta*ialpha + ibeta for imer in range(nmer) for in0 in range(nn0) for iT0 in range(nT0) for ialpha in range(4) for ibeta in range(1,10)]],
                          Htop_ice[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                            + nn0*nT0*nalpha*nbeta * imer
                                            + nT0*nalpha*nbeta * in0
                                            + nalpha*nbeta * iT0
                                            + nbeta*ialpha + ibeta for imer in range(nmer) for in0 in range(nn0) for iT0 in range(nT0) for ialpha in range(4) for ibeta in range(1,10)]] ])
                          
    under = len(np.nonzero(Lout<lim_under)[0])
    over = len(np.nonzero(Lout>lim_over)[0])
    
    if under/(nmer*nn0*nT0*4*9*3) >= 1:
        print(ievent, ivespa_ID[ievent], under/(nmer*nn0*nT0*4*9*3))
        list_under.append([ievent, ivespa_ID[ievent]])
        
    elif over/(nmer*nn0*nT0*4*9*3) >= 1:
        print(ievent, ivespa_ID[ievent], over/(nmer*nn0*nT0*4*9*3))
        list_over.append([ievent, ivespa_ID[ievent]])

excluded = list(np.array(np.array(list_under)[:,0], dtype=float)) + list(np.array(np.array(list_over)[:,0], dtype=float))

events = np.array(range(nevent))
events = events[np.nonzero([(event not in excluded) for event in events])[0]]
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_outliers[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_outliers[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_outliers[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_outliers[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% LOWEST DELTA_H/H

entr_DH = np.zeros((5, nmc, 2))
better_0D_DH = np.zeros((5, nmc))

dtype=[('index', int), ('delta_norm', float)]
event_ind = np.array([(i, delta_norm[i]) for i in range(130)], dtype=dtype)
event_ind = np.sort(event_ind, order='delta_norm')
events = event_ind['index']
events = events[:30]
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_DH[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_DH[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_DH[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_DH[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% Weak plumes

entr_weaks = np.zeros((5, nmc, 2))
better_0D_weaks = np.zeros((5, nmc))

weaks = np.nonzero(ivespa.Plume_Morphology[ivespa_indexes]=='weak')[0]
events = weaks
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_weaks[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_weaks[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_weaks[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_weaks[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% Strong plumes

entr_strongs = np.zeros((5, nmc, 2))
better_0D_strongs = np.zeros((5, nmc))

weaks = np.nonzero(ivespa.Plume_Morphology[ivespa_indexes]=='strong')[0]
events = weaks
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_strongs[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_strongs[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_strongs[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_strongs[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% Wet eruptions

entr_wet = np.zeros((5, nmc, 2))
better_0D_wet = np.zeros((5, nmc))

imer, in0, iT0 = 5, 2, 2
ia_dry, ib_dry = 3, 0
ia_ice, ib_ice = 4, 0
ia_0, ib_0 = 2, 0
ia_2, ib_2 = 5, 0
    
L_pl = []
L_ge = []
a = []
for ievent in range(nevent):
    L_pl.append((Htop_ice[nmer*nn0*nT0*nalpha*nbeta * ievent
                                           + nn0*nT0*nalpha*nbeta * imer
                                           + nT0*nalpha*nbeta * in0
                                           + nalpha*nbeta * iT0
                                           + nbeta*ia_ice + ib_ice]-Htop_dry[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                  + nn0*nT0*nalpha*nbeta * imer
                                                                                  + nT0*nalpha*nbeta * in0
                                                                                  + nalpha*nbeta * iT0
                                                                                  + nbeta*ia_dry + ib_dry]) / eru_h_avl[ievent]) 
                                                                             
    L_ge.append((out_Htop2[nmer*nn0*nT0 * ievent 
                                    + nn0*nT0 * imer
                                    + nT0 * in0
                                    + iT0, nbeta*ia_2+ib_2]-out_Htop0[nmer*nn0*nT0 * ievent 
                                                                    + nn0*nT0 * imer
                                                                    + nT0 * in0
                                                                    + iT0, nbeta*ia_0+ib_0]) / eru_h_avl[ievent] )
                                                                             
    a.append(MER[nmer*nn0*nT0*nalpha*nbeta * ievent
                                + nn0*nT0*nalpha*nbeta * imer
                                + nT0*nalpha*nbeta * in0
                                + nalpha*nbeta * iT0
                                + nbeta*ia_dry + ib_dry])
                                                                             
L_pl=np.array(L_pl)
L_ge=np.array(L_ge)
ivespa_ID=np.array(ivespa_ID)
lim_pl, lim_ge = 0.125, 0.85
event_ind = np.concatenate((np.nonzero(L_pl>lim_pl)[0], np.nonzero(L_ge>lim_ge)[0]), axis=0)
mer = np.array(a)[event_ind]
mmer = np.array(a)


dtype=[('index', int), ('height', float)]

event_ind=np.array([(event_ind[i], mer[i]) for i in range(len(event_ind))], dtype=dtype)

events = np.sort(event_ind, order='height')
events=events['index']
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_wet[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_wet[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_wet[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_wet[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% Dry eruptions

entr_dry = np.zeros((5, nmc, 2))
better_0D_dry = np.zeros((5, nmc))

imer, in0, iT0 = 5, 2, 2
ia_dry, ib_dry = 3, 0
ia_ice, ib_ice = 4, 0
ia_0, ib_0 = 2, 0
ia_2, ib_2 = 5, 0
    
L_pl = []
L_ge = []
a = []
for ievent in range(nevent):
    L_pl.append((Htop_ice[nmer*nn0*nT0*nalpha*nbeta * ievent
                                           + nn0*nT0*nalpha*nbeta * imer
                                           + nT0*nalpha*nbeta * in0
                                           + nalpha*nbeta * iT0
                                           + nbeta*ia_ice + ib_ice]-Htop_dry[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                  + nn0*nT0*nalpha*nbeta * imer
                                                                                  + nT0*nalpha*nbeta * in0
                                                                                  + nalpha*nbeta * iT0
                                                                                  + nbeta*ia_dry + ib_dry]) / eru_h_avl[ievent]) 
                                                                             
    L_ge.append((out_Htop2[nmer*nn0*nT0 * ievent 
                                    + nn0*nT0 * imer
                                    + nT0 * in0
                                    + iT0, nbeta*ia_2+ib_2]-out_Htop0[nmer*nn0*nT0 * ievent 
                                                                    + nn0*nT0 * imer
                                                                    + nT0 * in0
                                                                    + iT0, nbeta*ia_0+ib_0]) / eru_h_avl[ievent] )
                                                                             
    a.append(MER[nmer*nn0*nT0*nalpha*nbeta * ievent
                                + nn0*nT0*nalpha*nbeta * imer
                                + nT0*nalpha*nbeta * in0
                                + nalpha*nbeta * iT0
                                + nbeta*ia_dry + ib_dry])
                                                                             
L_pl=np.array(L_pl)
L_ge=np.array(L_ge)
ivespa_ID=np.array(ivespa_ID)
lim_pl, lim_ge = 0.005, 0.06
event_ind = np.concatenate((np.nonzero(abs(L_pl)<lim_pl)[0], np.nonzero(abs(L_ge)<lim_ge)[0]), axis=0)
mer = np.array(a)[event_ind]
mmer = np.array(a)

print(len(np.nonzero(abs(L_pl)<lim_pl)[0]))
print(len(np.nonzero(abs(L_ge)<lim_ge)[0]))

dtype=[('index', int), ('height', float)]

event_ind=np.array([(event_ind[i], mer[i]) for i in range(len(event_ind))], dtype=dtype)

events = np.sort(event_ind, order='height')
events=events['index']
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_dry[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_dry[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_dry[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_dry[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% High Durations excluded

entr_dur = np.zeros((5, nmc, 2))
better_0D_dur = np.zeros((5, nmc))

events = np.nonzero(0.2<eru_dur)[0]
events = np.nonzero([(0.2<dur<96) for dur in eru_dur])[0]
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_dur[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_dur[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_dur[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_dur[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
# %% IVESPA flags wheighting

entr_flags = np.zeros((5, nmc, 2))
better_0D_flags = np.zeros((5, nmc))

options = {
    'dtype': {'30': float, '32': float, '55': float, '56': float, '57': float, '58': float},
    'engine': 'openpyxl' 
}

ivtab = pd.read_excel(r"/Users/titouanrenaud/Documents/ENS/ARPE/ATMOSPHERIC PROFILES/IVESPAvn1.0.xlsx", **options,header=0).iloc[ivespa_indexes]

Htop_flags = ivtab['Tephra Plume Top Best Estimate Flag']
Htop_un_flags = ivtab['Tephra Plume Top Uncertainty Flag']
dur_flags = ivtab['Duration Best Estimate Flag']
dur_un_flags = ivtab['Duration Uncertainty Flag']
TEM_flags = ivtab['TEM Best Estimate Flag']
TEM_un_flags = ivtab['TEM Uncertainty Flag']

BE_flags = np.array([Htop_flags, dur_flags, TEM_flags])
UN_flags = np.array([Htop_un_flags, dur_un_flags, TEM_un_flags])
ALL_flags = np.array([Htop_flags, Htop_un_flags, dur_flags, dur_un_flags, TEM_flags, TEM_un_flags])

for a in range(np.shape(ALL_flags)[0]):
    for b in range(np.shape(ALL_flags)[1]):
        if ALL_flags[a,b] == 'Unknown':
            ALL_flags[a,b] = 2
            
        elif ALL_flags[a,b] == 'na':
            ALL_flags[a,b] = 2

events = np.nonzero([len(np.nonzero((ALL_flags[:,i]<2))[0])==6 for i in range(130)])[0]
ngood = len(events)
ngood

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_flags[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_flags[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_flags[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_flags[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% Tropical events

entr_trop = np.zeros((5, nmc, 2))
better_0D_trop = np.zeros((5, nmc))

events = np.nonzero(abs(ivespa.LatitudeN[ivespa_indexes])<18)[0]
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_trop[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_trop[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_trop[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_trop[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1

# %% High latitudes

entr_hl = np.zeros((5, nmc, 2))
better_0D_hl = np.zeros((5, nmc))

events = np.nonzero(abs(ivespa.LatitudeN[ivespa_indexes])>60.47)[0]
ngood = len(events)

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties

n = 2
i = 0

for neqn in range(3): 
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n)
                                      
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_hl[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                            + nn0*nT0*nalpha*nbeta * imer
                                                            + nT0*nalpha*nbeta * in0
                                                            + nalpha*nbeta * iT0
                                                            + nbeta*ialpha + ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
        
        better_0D_hl[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1
    
for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line = np.zeros((nalpha, nbeta))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line[ialpha, ibeta] = (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                                          + nn0*nT0 * imer
                                                                                          + nT0 * in0
                                                                                          + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n ))**(1/n)
         
        MIN = np.where(line==line.min())
        ialpha, ibeta = MIN[0][0], MIN[1][0]
        entr_hl[i, imc, :] = ialpha, ibeta

        better = (abs(rand_h2[events,imc] -outH[nmer*nn0*nT0 * events 
                                                    + nn0*nT0 * imer
                                                    + nT0 * in0
                                                    + iT0, nbeta*ialpha+ibeta]) < abs(rand_h2[events,imc] - a2[0]*merlist**a2[1]))
  
        better_0D_hl[i, imc] = len(np.nonzero(better)[0]) / ngood
        
    i += 1



# %% PLOT boxes 80-95

mod1D_used = [0,1,2,3,4] #[dry, wet, ice, 0.0, e-2]

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

with open(f'{dir_fig}/values.txt', 'w') as f:
    f.write('Values given in the boxplots (Figure 4):\n\n')

def plot_box(ax, frame, legend=False):
    positions = range(1, len(frame)+1)
    for pos, data in zip(positions, frame):
        quant80 = [np.percentile(data, 10), np.percentile(data, 90)]
        quant95 = [np.percentile(data, 2.5), np.percentile(data, 97.5)]
        l2 = ax.fill_between(quant80, pos-0.27, pos+0.27, color='lightsteelblue', alpha=0.6, linewidth=1.0, edgecolor='darkblue')
        l1, = ax.plot(quant95, [pos, pos], color='firebrick', linestyle='-', marker='|', markersize=30, linewidth=1.8, alpha=1, markeredgewidth=1.5)        
        mean, med = np.mean(data), np.median(data)
        l3 = ax.scatter(mean, pos, marker='o', s=140, color='firebrick', alpha=0.9)
        l4 = ax.scatter(med, pos, marker='^', s=290, color='darkviolet', edgecolor='darkviolet', alpha=0.5)

        if legend:
            ax.legend(handles=[l2, l1, l3, l4], labels=['80\% quantiles', '95\% quantiles', 'mean', 'median'], fontsize=18, loc=(0.57, 0.7), 
                      frameon=True, fancybox=True, handletextpad=0.7, borderpad=0.4, handlelength=2.3, numpoints=2, scatterpoints=1, labelspacing=0.7)

with plt.style.context(["science"]): 

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(20,12), layout='constrained', width_ratios=[0.35, 0.35, 0.3])
    
    ### panel 1: beta
    ax[0].set_title(r'$\textbf{a)}$ Most performant $\beta$ value', fontsize=22)
    ax[0].yaxis.set_inverted(True)
    ax[0].set_xlim(0, 1)
    ax[0].set_xticks(betalist, labels=[round(beta,3) for beta in betalist], minor=False, fontsize=17)
    ax[0].tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
    labels=['All (n=130)', u'Outliers excluded (n=124)', u'Extreme durations \n excluded (n=105)', u'Lowest relative \n uncertainties (n=30)', 'Weak plumes (n=22)', 'Strong plumes (n=22)', 'Wet (n=30)', 'Dry (n=30)','Tropical (n=30)', 'High latitudes (n=40)', u'Data interpretation \n flags 2 excluded (n=21)']
    miny, maxy = -0.8, len(labels)+0.5
    ax[0].set_ylim(maxy, miny)
    ax[0].set_yticks(range(len(labels)), labels=labels, minor=False, fontsize=14)
    ax[0].grid(which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.5)

    ax[0].plot([0.35, 1.0], [0,0], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[0].text(0.675, 0.1, r'$\textit{Lab experiments}$', fontsize=18, ha='center', va='top')
    ax[0].plot([0.35, 0.35], [0, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax[0].plot([0.1, 0.3], [0.4,0.4], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[0].text(0.2, 0.3, r'$\textit{3D SK2015}$', fontsize=16, ha='center', va='bottom')
    ax[0].plot([0.1, 0.1], [0.4, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax[0].plot([0.3, 0.3], [0.4, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax[0].plot([0.2, 0.4], [-0.3,-0.3], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[0].text(0.3, -0.4, r'$\textit{3D this study}$', fontsize=18, ha='center', va='bottom')
    ax[0].plot([0.2, 0.2], [-0.3, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax[0].plot([0.4, 0.4], [-0.3, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    beta_distri_all = np.array(betalist)[np.array(np.concatenate(entr_all[mod1D_used, :, 1]), dtype=int)]
    beta_distri_outliers = np.array(betalist)[np.array(np.concatenate(entr_outliers[mod1D_used, :, 1]), dtype=int)]
    beta_distri_DH = np.array(betalist)[np.array(np.concatenate(entr_DH[mod1D_used, :, 1]), dtype=int)]
    beta_distri_weaks = np.array(betalist)[np.array(np.concatenate(entr_weaks[mod1D_used, :, 1]), dtype=int)]
    beta_distri_strongs = np.array(betalist)[np.array(np.concatenate(entr_strongs[mod1D_used, :, 1]), dtype=int)]
    beta_distri_wet = np.array(betalist)[np.array(np.concatenate(entr_wet[mod1D_used, :, 1]), dtype=int)]
    beta_distri_dry = np.array(betalist)[np.array(np.concatenate(entr_dry[mod1D_used, :, 1]), dtype=int)]
    beta_distri_dur = np.array(betalist)[np.array(np.concatenate(entr_dur[mod1D_used, :, 1]), dtype=int)]
    beta_distri_flags = np.array(betalist)[np.array(np.concatenate(entr_flags[mod1D_used, :, 1]), dtype=int)]
    beta_distri_trop = np.array(betalist)[np.array(np.concatenate(entr_trop[mod1D_used, :, 1]), dtype=int)]
    beta_distri_hl = np.array(betalist)[np.array(np.concatenate(entr_hl[mod1D_used, :, 1]), dtype=int)]

    plot_box(ax[0], [beta_distri_all, beta_distri_outliers, beta_distri_dur, beta_distri_DH, beta_distri_weaks, beta_distri_strongs, beta_distri_wet, beta_distri_dry, beta_distri_trop, beta_distri_hl, beta_distri_flags], legend=True)

    with open(f'{dir_fig}/values.txt', 'a') as f:
        f.write('** Mean and median values for beta:\n\n' +
                f'All: mean={np.mean(beta_distri_all):.3f}, median={np.median(beta_distri_all):.3f}\n' +
                f'Outliers excluded: mean={np.mean(beta_distri_outliers):.3f}, median={np.median(beta_distri_outliers):.3f}\n' +
                f'Extreme durations excluded: mean={np.mean(beta_distri_dur):.3f}, median={np.median(beta_distri_dur):.3f}\n' +
                f'Lowest relative uncertainties: mean={np.mean(beta_distri_DH):.3f}, median={np.median(beta_distri_DH):.3f}\n' +
                f'Weak plumes: mean={np.mean(beta_distri_weaks):.3f}, median={np.median(beta_distri_weaks):.3f}\n' +
                f'Strong plumes: mean={np.mean(beta_distri_strongs):.3f}, median={np.median(beta_distri_strongs):.3f}\n' +
                f'Wet: mean={np.mean(beta_distri_wet):.3f}, median={np.median(beta_distri_wet):.3f}\n' +
                f'Dry: mean={np.mean(beta_distri_dry):.3f}, median={np.median(beta_distri_dry):.3f}\n' +
                f'Tropical: mean={np.mean(beta_distri_trop):.3f}, median={np.median(beta_distri_trop):.3f}\n' +
                f'High latitudes: mean={np.mean(beta_distri_hl):.3f}, median={np.median(beta_distri_hl):.3f}\n' +
                f'Data interpretation flags 2 excluded: mean={np.mean(beta_distri_flags):.3f}, median={np.median(beta_distri_flags):.3f}\n\n')

    ### panel 2: probability 1D better than 0D
    ax[2].set_title(r'$\textbf{c)}$ Probability error 1D smaller than error 0D', fontsize=22)
    ax[2].yaxis.set_inverted(True)
    ax[2].set_xlim(0.29, 0.62)
    ax[2].set_xticks([0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], labels=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], minor=False, fontsize=17)
    ax[2].tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
    ax[2].fill_between([-0.5, 0.5], [miny, miny], [maxy, maxy], color='grey', alpha=0.2, linewidth=0)
    ax[2].fill_between([0.5, 1.5], [miny, miny], [maxy, maxy], color='grey', alpha=0.03, linewidth=0)
    ax[2].grid(which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.5)
    
    ax[2].text((0.29+0.5)/2, 0.2, r'$\textit{0D better}$', fontsize=20, ha='center')
    ax[2].text((0.5+0.62)/2, 0.2, r'$\textit{1D better}$', fontsize=20, ha='center')
    
    distri_all = np.concatenate(better_0D_all[mod1D_used, :])
    distri_outliers = np.concatenate(better_0D_outliers[mod1D_used, :])
    distri_DH = np.concatenate(better_0D_DH[mod1D_used, :])
    distri_weaks = np.concatenate(better_0D_weaks[mod1D_used, :])
    distri_strongs = np.concatenate(better_0D_strongs[mod1D_used, :])
    distri_wet = np.concatenate(better_0D_wet[mod1D_used, :])
    distri_dry = np.concatenate(better_0D_dry[mod1D_used, :])
    distri_dur = np.concatenate(better_0D_dur[mod1D_used, :])
    distri_flags = np.concatenate(better_0D_flags[mod1D_used, :])
    distri_trop = np.concatenate(better_0D_trop[mod1D_used, :])
    distri_hl = np.concatenate(better_0D_hl[mod1D_used, :])
    
    markers = ['o', 'x', 'v', 's', '*']
    labells = ['dry Plumerise', 'wet Plumerise', 'ice Plumerise', 'dry Geneva', 'wet Geneva']
    colors = ['#264653', 'purple', '#2a9d8f', '#bc6c25', '#e76f51']

    with open(f'{dir_fig}/values.txt', 'a') as f:
        f.write('** Probability error lower than 0D Empirical:\n\n')
                
    for i in range(len(mod1D_used)):
        with open(f'{dir_fig}/values.txt', 'a') as f:
            f.write(f'Model = {labells[i]}: \n' + 
                    f'All: {np.mean(better_0D_all[i, :]):.3f}\n' + 
                    f'Outliers excluded: {np.mean(better_0D_outliers[i, :]):.3f}\n' +
                    f'Extreme durations excluded: {np.mean(better_0D_dur[i, :]):.3f}\n' +
                    f'Lowest relative uncertainties: {np.mean(better_0D_DH[i, :]):.3f}\n' +
                    f'Weak plumes: {np.mean(better_0D_weaks[i, :]):.3f}\n' +
                    f'Strong plumes: {np.mean(better_0D_strongs[i, :]):.3f}\n' +
                    f'Wet: {np.mean(better_0D_wet[i, :]):.3f}\n' +
                    f'Dry: {np.mean(better_0D_dry[i, :]):.3f}\n' +
                    f'Tropical: {np.mean(better_0D_trop[i, :]):.3f}\n' +
                    f'High latitudes: {np.mean(better_0D_hl[i, :]):.3f}\n' +
                    f'Data interpretation flags 2 excluded: {np.mean(better_0D_flags[i, :]):.3f}\n\n')
            
        ax[2].plot([np.mean(better_0D_all[i, :]), np.mean(better_0D_outliers[i, :]), np.mean(better_0D_dur[i, :]), np.mean(better_0D_DH[i, :]), np.mean(better_0D_weaks[i, :]), np.mean(better_0D_strongs[i, :]), np.mean(better_0D_wet[i, :]), np.mean(better_0D_dry[i, :]), np.mean(better_0D_trop[i, :]), np.mean(better_0D_hl[i, :]), np.mean(better_0D_flags[i, :])],
                   [1,2,3,4,5,6,7, 8, 9, 10, 11], linestyle='None', marker=markers[i], markersize=8, label=labells[i], alpha=0.7, color=colors[i])
    
    ax[2].legend(fontsize=18, loc=(0.03, 0.69), frameon=True, fancybox=True, handletextpad=0.1, borderpad=0.3, handlelength=1.4)

    # ### panel 3: alpha
    ax[1].set_title(r'$\textbf{b)}$ Most performant $\alpha$ value', fontsize=22)
    ax[1].yaxis.set_inverted(True)
    ax[1].set_xlim(0.04, 0.21)
    ax[1].set_xticks(alphalist, labels=alphalist, minor=False, fontsize=17)
    ax[1].tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)

    ax[1].plot([0.07, 0.15], [0,0], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[1].text((0.07+0.15)/2, 0.1, r'$\textit{Lab experiments}$', fontsize=18, ha='center', va='top')
    ax[1].plot([0.07, 0.07], [0, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax[1].plot([0.15, 0.15], [0, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax[1].plot([0.05, 0.06], [0.4,0.4], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[1].text((0.05+0.06)/2, 0.3, r'$\textit{SK2015}$', fontsize=16, ha='center', va='bottom')
    ax[1].plot([0.05, 0.05], [0.4, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax[1].plot([0.06, 0.06], [0.4, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    

    ax[1].plot([0.108, 0.116], [-0.3,-0.3], color='k', linestyle='-', linewidth=0.8, marker='|', markersize=15, alpha=0.8)
    ax[1].text(0.112, -0.4, r'$\textit{3D this study}$', fontsize=18, ha='center', va='bottom')
    ax[1].plot([0.108, 0.108], [-0.3, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax[1].plot([0.116, 0.116], [-0.3, maxy], color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax[1].grid(which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.5)
    
    alpha_distri_all = np.array(alphalist)[np.array(np.concatenate(entr_all[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_outliers = np.array(alphalist)[np.array(np.concatenate(entr_outliers[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_DH = np.array(alphalist)[np.array(np.concatenate(entr_DH[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_weaks = np.array(alphalist)[np.array(np.concatenate(entr_weaks[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_strongs = np.array(alphalist)[np.array(np.concatenate(entr_strongs[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_wet = np.array(alphalist)[np.array(np.concatenate(entr_wet[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_dry = np.array(alphalist)[np.array(np.concatenate(entr_dry[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_dur = np.array(alphalist)[np.array(np.concatenate(entr_dur[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_flags = np.array(alphalist)[np.array(np.concatenate(entr_flags[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_trop = np.array(alphalist)[np.array(np.concatenate(entr_trop[mod1D_used, :, 0]), dtype=int)]
    alpha_distri_hl = np.array(alphalist)[np.array(np.concatenate(entr_hl[mod1D_used, :, 0]), dtype=int)]
    
    plot_box(ax[1], [alpha_distri_all, alpha_distri_outliers, alpha_distri_dur, alpha_distri_DH, alpha_distri_weaks, alpha_distri_strongs, alpha_distri_wet, alpha_distri_dry, alpha_distri_trop, alpha_distri_hl, alpha_distri_flags])

    with open(f'{dir_fig}/values.txt', 'a') as f:
        f.write('** Mean and median values for alpha:\n\n' +
                f'All: mean={np.mean(alpha_distri_all):.3f}, median={np.median(alpha_distri_all):.3f}\n' +
                f'Outliers excluded: mean={np.mean(alpha_distri_outliers):.3f}, median={np.median(alpha_distri_outliers):.3f}\n' +
                f'Extreme durations excluded: mean={np.mean(alpha_distri_dur):.3f}, median={np.median(alpha_distri_dur):.3f}\n' +
                f'Lowest relative uncertainties: mean={np.mean(alpha_distri_DH):.3f}, median={np.median(alpha_distri_DH):.3f}\n' +
                f'Weak plumes: mean={np.mean(alpha_distri_weaks):.3f}, median={np.median(alpha_distri_weaks):.3f}\n' +
                f'Strong plumes: mean={np.mean(alpha_distri_strongs):.3f}, median={np.median(alpha_distri_strongs):.3f}\n' +
                f'Wet: mean={np.mean(alpha_distri_wet):.3f}, median={np.median(alpha_distri_wet):.3f}\n' +
                f'Dry: mean={np.mean(alpha_distri_dry):.3f}, median={np.median(alpha_distri_dry):.3f}\n' +
                f'Tropical: mean={np.mean(alpha_distri_trop):.3f}, median={np.median(alpha_distri_trop):.3f}\n' +
                f'High latitudes: mean={np.mean(alpha_distri_hl):.3f}, median={np.median(alpha_distri_hl):.3f}\n' +
                f'Data interpretation flags 2 excluded: mean={np.mean(alpha_distri_flags):.3f}, median={np.median(alpha_distri_flags):.3f}\n\n')

    ax[0].set_yticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11], labels, fontsize=20)

    # plt.savefig(f'{dir_fig}/Figure4.pdf', dpi=300, format='pdf')
    # plt.close()

    plt.show()


# %%

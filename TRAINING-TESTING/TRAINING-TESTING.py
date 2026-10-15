#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:57:06 2025

@author: titouanrenaud
"""

import os
import numpy as np
import pandas as pd
import plumerise as pr
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import scienceplots
from matplotlib import rc
import scipy.stats as stats
from utils import load_input, getIVESPAtable, getAtmArray

init_dir = load_input() #Drive data folder

dir_fig = f'{init_dir}/TRAINING-TESTING/' #directory to save figures

outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs' #where to find models' outputs 

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

ivespa = getIVESPAtable() #uploading IVESPA table

ivespa_ID = list(outputs.ivespa_ID[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                for ievent in range(nevent)]])

ivespa_indexes = [list(ivespa['Ivespa_ID']).index(event) for event in ivespa_ID]

eru_h = np.array(ivespa['TopHeight'][ivespa_indexes])                   # Plume top height [km a.s.l]
eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][ivespa_indexes])     # Plume top height uncertainty
eru_vh = np.array(ivespa['VentAltitude'][ivespa_indexes])/1000          # Vent altitude [m a.s.l]
eru_h_avl = eru_h - eru_vh
eru_dur = np.array(ivespa['Duration'][ivespa_indexes])

save_directory = f'{outputs_dir}/MATLAB/L=0.0/' #uploading Geneva lambda=0.0 outputs
out_Htop0 = np.array(pd.read_csv(save_directory+'out_Htop.csv', low_memory = False, header=None))

save_directory = f'{outputs_dir}/MATLAB/L=1e-2/' #uploading Geneva lambda=1e-2 outputs
out_Htop2 = np.array(pd.read_csv(save_directory+'out_Htop.csv', low_memory = False, header=None))

prop_training=0.6   #proportion of events on which training will be conducted. Other events for testing
nb_try = 100        #number of training-testing samplings. Each time, a different random selection of events for training and testing will be made.

# %% TRAINING AND TESTING SAMPLINGS AMONG ALL EVENTS

events = np.array(range(nevent)) #defining the list of events taken. Here all 130 events, no selection.
ngood = nevent #number of events in the selection. Here 130

# %% TRAINING & TESTING FOR nb_try 

best_ab = np.zeros((nb_try,5,2))
table_performances = np.zeros((nb_try, 7, 7))

nmc = 1000 #10 000 Monte Carlo simulations. 
        
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
all_events = list(events)

for test in range(nb_try):
    
    training_set = random.sample(all_events, int(prop_training*nevent))
    testing_set = np.array(all_events)[np.nonzero([i not in training_set for i in all_events])[0]]
    
    # np.savetxt(dir_fig+f'RUN_{test} training, {eqn} Plumerise.csv', training_set, delimiter=",")
    # np.savetxt(dir_fig+f'RUN_{test} testing, {eqn} Plumerise.csv', testing_set, delimiter=",")
    
    with open(dir_fig+f"RUN_{test}.txt", "w") as fichier:
        fichier.write(f"RUN_{test}, {int(prop_training*100)}/{int(100-prop_training*100)} (training={len(training_set)} / testing={len(testing_set)} events)\n\n")
    
    
    ### Training the models ###
    
    ### Empirical power laws

    merlist = MER[[nmer*nn0*nT0*nalpha*nbeta * ievent + nn0*nT0*nalpha*nbeta * 5 for ievent in training_set]]
    
    def func(MER,a,b):
        return a * MER**b
    
    sigma=np.array(eru_d_h[training_set]/2, dtype=float)
    
    a2, b2 = curve_fit(func, merlist, eru_h_avl[training_set], sigma=sigma) #power-fit accounting for uncertainties
    a3, b3 = np.polyfit(np.log10(merlist), np.log10(np.array(eru_h_avl[training_set], dtype=float)), deg=1, w=sigma) #linear-fit in log-space accountign for uncertainties
    
    xdata = np.logspace(np.log10(min(merlist)), np.log10(max(merlist)), 1000)
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"poly-fit in linear, H = {round(a2[0],3)} MER ** {round(a2[1],3)}\n\n")
        fichier.write(f"linear-fit in log, H = {round(10**b3,3)} MER ** {round(a3,3)}\n\n")
    
    ### 1D models
    
    n = 2 # used in definition of RMS. 
    
    ### Plumerise models
    i=0
    for neqn in range(3): 
        
        eqn = ['dry', 'wet', 'ice'][neqn]
        Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
        
        rms_norm = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
        def func_rms(imc): #calculating RMS. (observed height-predicted height)^2/DELTA_H^2 for all events. 1 value per Monte Carlo simulation
            in0 = n0_indexes[imc]
            iT0 = T0_indexes[imc]
            imer = mer_indexes[imc]
            
            line=[]
            for ialpha in range(nalpha):
                for ibeta in range(nbeta):
                    line.append( (1/ngood * sum((abs(rand_h2[training_set,imc] - Htop[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                                       + nn0*nT0*nalpha*nbeta * imer
                                                                                                       + nT0*nalpha*nbeta * in0
                                                                                                       + nalpha*nbeta * iT0
                                                                                                       + nbeta*ialpha + ibeta for ievent in training_set]]) /
                                                   delta_h[[ievent for ievent in training_set]])**n ))**(1/n))
                                                  
            return line
        
        out_rms = Parallel(n_jobs=-1)(
            delayed(func_rms)(imc)
            for imc in range(nmc)
        )
        
        rms_norm = np.array(out_rms)
        
        heatmap = np.sum(rms_norm[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
        heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
        
        MIN1 = np.where(heatmap[:, :]==heatmap[:, :].min()) #(alpha, beta) couple minimizing the error
        
        alpha_star = nalpha-1-MIN1[0][0]
        beta_star = MIN1[1][0]
        
        best_ab[test, i, :] = [alpha_star, beta_star]
            
        with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
            fichier.write(f"{eqn} Plumerise, alpha*={alphalist[alpha_star]}, beta*={betalist[beta_star]}\n\n")

        i += 1   
            
    ### Geneva models
    for la in range(2): 
        val = ['0.0', '1e-2'][la]
        outH = [out_Htop0, out_Htop2][la]
        
        rms_norm = np.zeros((nalpha*nbeta, nmc))
        def func_rms(imc):
            in0 = n0_indexes[imc]
            iT0 = T0_indexes[imc]
            imer = mer_indexes[imc]
            line=[]
            for ialpha in range(nalpha):
                for ibeta in range(nbeta):
                    line.append( (1/ngood * sum((abs(rand_h2[training_set,imc] - outH[[nmer*nn0*nT0 * ievent 
                                                                      + nn0*nT0 * imer
                                                                      + nT0 * in0
                                                                      + iT0 for ievent in training_set],nbeta*ialpha+ibeta]) /
                                                   delta_h[[ievent for ievent in training_set]])**n))**(1/n))
                    
            return line
        
        out = Parallel(n_jobs=-1)(
            delayed(func_rms)(imc)
            for imc in range(nmc)
        )
        
        rms_norm = np.array(out)
        
        heatmap = np.sum(rms_norm[:,:], axis=0)/nmc
        heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
        
        MIN1 = np.where(heatmap[:, :]==heatmap[:, :].min())
        
        alpha_star = nalpha-1-MIN1[0][0]
        beta_star = MIN1[1][0]
        
        best_ab[test, i,:] = [alpha_star, beta_star]
            
        with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
            fichier.write(f"Geneva lambda= {val}, alpha*={alphalist[alpha_star]}, beta*={betalist[beta_star]}\n\n")

        i += 1  
    
    
    ### TESTING ###
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write("TESTING\n\n")
        
    ### Plumerise
    def func_rms(imc, Htop): #calculating RMS. (observed height-predicted height)^2/DELTA_H^2 for all events. 1 value per Monte Carlo simulation
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line=[]
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line.append( (1/ngood * sum((abs(rand_h2[testing_set,imc] - Htop[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta for ievent in testing_set]]) /
                                               delta_h[[ievent for ievent in testing_set]])**n ))**(1/n))
                                              
        return line
    
    ### dry Plumerise
    rms_norm_dry = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, Htop_dry)
        for imc in range(nmc)
    )
    
    rms_norm_dry = np.array(out_rms)
    
    heatmap = np.sum(rms_norm_dry[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    err = heatmap[nalpha-1-int(best_ab[test,0][0]), int(best_ab[test,0][1])]
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"dry Plumerise, err = {err}\n\n")
        
        
    ### wet Plumerise
    rms_norm_wet = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, Htop_wet)
        for imc in range(nmc)
    )
    
    rms_norm_wet = np.array(out_rms)
    
    heatmap = np.sum(rms_norm_wet[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    err = heatmap[nalpha-1-int(best_ab[test,1][0]), int(best_ab[test,1][1])]
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"wet Plumerise, err = {err}\n\n")
        
    prop_better = len(np.nonzero(rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,1,0] = prop_better
    
    prop_worse = len(np.nonzero(rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,1] = prop_worse
    
    
    ### ice Plumerise
    rms_norm_ice = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, Htop_ice)
        for imc in range(nmc)
    )
    
    rms_norm_ice = np.array(out_rms)
    
    heatmap = np.sum(rms_norm_ice[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    err = heatmap[nalpha-1-int(best_ab[test,2][0]), int(best_ab[test,2][1])]
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"ice Plumerise, err = {err}\n\n")
        
    prop_better_dry = len(np.nonzero(rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + 
                                                  int(best_ab[test,2][1])] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta 
                                                                                          + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,2,0] = prop_better_dry
    
    prop_worse_dry = len(np.nonzero(rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + 
                                                  int(best_ab[test,2][1])] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta 
                                                                                          + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,2] = prop_worse_dry
    
    prop_better_wet = len(np.nonzero(rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])] < rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,2,1] = prop_better_wet
    
    prop_worse_wet = len(np.nonzero(rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])] > rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,1,2] = prop_worse_wet
    
    ### Geneva model
    def func_rms(imc, outH):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        line=[]
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line.append( (1/ngood * sum((abs(rand_h2[testing_set,imc] - outH[[nmer*nn0*nT0 * ievent 
                                                                  + nn0*nT0 * imer
                                                                  + nT0 * in0
                                                                  + iT0 for ievent in testing_set],nbeta*ialpha+ibeta]) /
                                               delta_h[[ievent for ievent in testing_set]])**n))**(1/n))
                
        return line
    
    ### Geneva 0.0
    rms_norm_0 = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, out_Htop0)
        for imc in range(nmc)
    )
    
    rms_norm_0 = np.array(out_rms)
    
    heatmap = np.sum(rms_norm_0[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    err = heatmap[nalpha-1-int(best_ab[test,3][0]), int(best_ab[test,3][1])]
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"Geneva lambda=0.0, err = {err}\n\n")
        
    prop_better_dry = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,3,0] = prop_better_dry
    
    prop_worse_dry = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,3] = prop_worse_dry
    
    prop_better_wet = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] < rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,3,1] = prop_better_wet
    
    prop_worse_wet = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] > rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,1,3] = prop_worse_wet
            
    prop_better_ice = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] < rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,3,2] = prop_better_ice
    
    prop_worse_ice = len(np.nonzero(rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])] > rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,2,3] = prop_worse_ice

    
    ### Geneva 1e-2
    rms_norm_2 = np.zeros((nalpha*nbeta, nmc)) #RMS matrix
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, out_Htop2)
        for imc in range(nmc)
    )
    
    rms_norm_2 = np.array(out_rms)
    
    heatmap = np.sum(rms_norm_2[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    err = heatmap[nalpha-1-int(best_ab[test,4][0]), int(best_ab[test,4][1])]
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"Geneva lambda=1e-2, err = {err}\n\n")
        
    prop_better_dry = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,4,0] = prop_better_dry
    
    prop_worse_dry = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,4] = prop_worse_dry
    
    prop_better_wet = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] < rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,4,1] = prop_better_wet
    
    prop_worse_wet = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] > rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,1,4] = prop_worse_wet
    
    prop_better_ice = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] < rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,4,2] = prop_better_ice
    
    prop_worse_ice = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] > rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,2,4] = prop_worse_ice
    
    prop_better_0 = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] < rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,4,3] = prop_better_0
    
    prop_worse_0 = len(np.nonzero(rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])] > rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,3,4] = prop_worse_0
    
    
    ### Empirical power-laws
    def func_rms(imc, a, b):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        line=[]
        line.append( (1/ngood * sum((abs(rand_h2[testing_set,imc] - a*MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                           + nn0*nT0*nalpha*nbeta * imer
                                                                                           + nT0*nalpha*nbeta * in0
                                                                                           + nalpha*nbeta * iT0 for ievent in testing_set]]**b)**n/
                                       ((delta_h[[ievent for ievent in testing_set]])**n ))))**(1/n))
                
        return line
    
    ### fit in linear    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, a2[0], a2[1])
        for imc in range(nmc)
    )
    
    rms_norm_linear = np.array(out_rms)
    
    err = np.sum(rms_norm_linear[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"Poly-fit in linear space, err = {err[0]}\n\n")
        
    prop_better_dry = len(np.nonzero(rms_norm_linear[:,0] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,5,0] = prop_better_dry
    
    prop_worse_dry = len(np.nonzero(rms_norm_linear[:,0] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,5] = prop_worse_dry
    
    prop_better_wet = len(np.nonzero(rms_norm_linear[:,0] < rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,5,1] = prop_better_wet
    
    prop_worse_wet = len(np.nonzero(rms_norm_linear[:,0] > rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,1,5] = prop_worse_wet
    
    prop_better_ice = len(np.nonzero(rms_norm_linear[:,0] < rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,5,2] = prop_better_ice
    
    prop_worse_ice = len(np.nonzero(rms_norm_linear[:,0] > rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,2,5] = prop_worse_ice
    
    prop_better_0 = len(np.nonzero(rms_norm_linear[:,0] < rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,5,3] = prop_better_0
    
    prop_worse_0 = len(np.nonzero(rms_norm_linear[:,0] > rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,3,5] = prop_worse_0
    
    prop_better_2 = len(np.nonzero(rms_norm_linear[:,0] < rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])])[0])/nmc
    table_performances[test,5,4] = prop_better_2
    
    prop_worse_2 = len(np.nonzero(rms_norm_linear[:,0] > rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])])[0])/nmc
    table_performances[test,4,5] = prop_worse_2
    
    ### fit in log
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc, 10**b3, a3)
        for imc in range(nmc)
    )
    
    rms_norm_log = np.array(out_rms)
    
    err = np.sum(rms_norm_log[:,:], axis=0)/nmc #sum over the nmc Monte Carlo simulation. In the end one value of error per (alpha, beta) couple, averaged over alla MC simulations ond all ngood events.
    
    with open(dir_fig+f"RUN_{test}.txt", "a") as fichier:
        fichier.write(f"Linear-fit in log space, err = {err[0]}\n\n")
        
    prop_better_dry = len(np.nonzero(rms_norm_log[:,0] < rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,6,0] = prop_better_dry
    
    prop_worse_dry = len(np.nonzero(rms_norm_log[:,0] > rms_norm_dry[:,(nalpha-1-int(best_ab[test,0][0]))*nbeta + int(best_ab[test,0][1])])[0])/nmc
    table_performances[test,0,6] = prop_worse_dry
    
    prop_better_wet = len(np.nonzero(rms_norm_log[:,0] < rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,6,1] = prop_better_wet
    
    prop_worse_wet = len(np.nonzero(rms_norm_log[:,0] > rms_norm_wet[:,(nalpha-1-int(best_ab[test,1][0]))*nbeta + int(best_ab[test,1][1])])[0])/nmc
    table_performances[test,1,6] = prop_worse_wet
    
    prop_better_ice = len(np.nonzero(rms_norm_log[:,0] < rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,6,2] = prop_better_ice
    
    prop_worse_ice = len(np.nonzero(rms_norm_log[:,0] > rms_norm_ice[:,(nalpha-1-int(best_ab[test,2][0]))*nbeta + int(best_ab[test,2][1])])[0])/nmc
    table_performances[test,2,6] = prop_worse_ice
    
    prop_better_0 = len(np.nonzero(rms_norm_log[:,0] < rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,6,3] = prop_better_0
    
    prop_worse_0 = len(np.nonzero(rms_norm_log[:,0] > rms_norm_0[:,(nalpha-1-int(best_ab[test,3][0]))*nbeta + int(best_ab[test,3][1])])[0])/nmc
    table_performances[test,3,6] = prop_worse_0
    
    prop_better_2 = len(np.nonzero(rms_norm_log[:,0] < rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])])[0])/nmc
    table_performances[test,6,4] = prop_better_2
    
    prop_worse_2 = len(np.nonzero(rms_norm_log[:,0] > rms_norm_2[:,(nalpha-1-int(best_ab[test,4][0]))*nbeta + int(best_ab[test,4][1])])[0])/nmc
    table_performances[test,4,6] = prop_worse_2
    
    prop_better_linear = len(np.nonzero(rms_norm_log[:,0] < rms_norm_linear[:,0])[0])/nmc
    table_performances[test,6,5] = prop_better_linear
    
    prop_worse_linear = len(np.nonzero(rms_norm_log[:,0] > rms_norm_linear[:,0])[0])/nmc
    table_performances[test,5,6] = prop_worse_linear
    
    ### save
    np.savetxt(dir_fig+f'RUN_{test} best_entrainment.csv', best_ab[test,:,:], delimiter=",")
    np.savetxt(dir_fig+f'RUN_{test} relative_performances.csv', table_performances[test,:,:], delimiter=",")

table_moy = np.mean(table_performances, 0)
np.savetxt(dir_fig+'table_mean.csv', table_moy, delimiter=",")

table_sigma = table_performances.std(ddof=1, axis=0)
table_ci = stats.t.interval(0.95, df=nb_try-1, loc=table_moy, scale=table_sigma / np.sqrt(nb_try))

table_ci[0][np.nonzero(table_ci[0]<0)]=0
table_ci[1][np.nonzero(table_ci[1]>1)]=1

np.savetxt(dir_fig+'table_95_lower.csv', table_ci[0], delimiter=",")
np.savetxt(dir_fig+'table_95_upper.csv', table_ci[1], delimiter=",")


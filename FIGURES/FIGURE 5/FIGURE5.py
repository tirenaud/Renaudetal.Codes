#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 19:00:04 2025

@author: titouanrenaud
"""
# %% imports

from mimetypes import init
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
from utils import load_input

init_dir = load_input() #Drive data folder

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

# %% Condition Plumerise & Geneva 

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
# lim_pl, lim_ge = 0.085, 0.6
lim_pl, lim_ge = 0.125, 0.85
event_ind = np.concatenate((np.nonzero(L_pl>lim_pl)[0], np.nonzero(L_ge>lim_ge)[0]), axis=0)
if 69 not in event_ind: # adding MERAPI 2010
    event_ind = np.append(event_ind, 69)
mer = np.array(a)[event_ind]
mmer = np.array(a)

dtype=[('index', int), ('height', float)]

event_ind=np.array([(event_ind[i], mer[i]) for i in range(len(event_ind))], dtype=dtype)

events = np.sort(event_ind, order='height')
events=events['index']
ngood = len(events)

# %% Models tuned against full database

best_entrainment = np.array(pd.read_csv(f'{init_dir}/BEST_ENTRAINMENTS_ALL_EVENTS.csv', delimiter=';', header=None))
linfit = [0.176, 0.262]

error_all = []

entrainment = np.array(best_entrainment)
i = 0
n = 2
for neqn in range(3):
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    ialpha, ibeta = entrainment[i]
    line=[]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]  
        
        line.append( (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                           + nn0*nT0*nalpha*nbeta * imer
                                                                                           + nT0*nalpha*nbeta * in0
                                                                                           + nalpha*nbeta * iT0
                                                                                           + nbeta*ialpha + ibeta]) /
                                       delta_h[events])**n ))**(1/n))
    
    error_all.append((round(np.quantile(line, 0.05),1), round(np.mean(line),1), round(np.quantile(line, 0.95),1)))
    i+=1


for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    ialpha, ibeta = entrainment[i]
    line=[]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]  
    
        line.append( (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                          + nn0*nT0 * imer
                                                          + nT0 * in0
                                                          + iT0, nbeta*ialpha+ibeta]) /
                                       delta_h[events])**n))**(1/n))
        
    error_all.append((round(np.quantile(line, 0.05),1), round(np.mean(line),1), round(np.quantile(line, 0.95),1)))
    i+=1

for imc in range(nmc):
    in0 = n0_indexes[imc]
    iT0 = T0_indexes[imc]
    imer = mer_indexes[imc]  

    line.append( (1/ngood * sum((abs(rand_h2[events,imc] - linfit[0]*MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                       + nn0*nT0*nalpha*nbeta * imer
                                                                                       + nT0*nalpha*nbeta * in0
                                                                                       + nalpha*nbeta * iT0 for ievent in events]]**linfit[1]) /
                                   delta_h[events])**n))**(1/n))
    
error_all.append((round(np.quantile(line, 0.05),1), round(np.mean(line),1), round(np.quantile(line, 0.95),1)))

# %% Models tuned against wet eruptions

# %%% Empirical power laws

merlist = MER[nmer*nn0*nT0*nalpha*nbeta * events + nn0*nT0*nalpha*nbeta * 5 + nT0*nalpha*nbeta * 2 + nalpha*nbeta * 3]

def func(MER,a,b):
    return a * MER**b

sigma=np.array(eru_d_h[events]/2, dtype=float)

a2, b2 = curve_fit(func, merlist, eru_h_avl[events], sigma=sigma) #power-fit accounting for uncertainties
a3, b3 = np.polyfit(np.log10(merlist), np.log10(np.array(eru_h_avl[events], dtype=float)), deg=1, w=sigma) #linear-fit in log-space accountign for uncertainties

xdata = np.logspace(np.log10(min(merlist)), np.log10(max(merlist)), 1000)

# %%% 1D models

best_entrainment_wet = np.zeros((5,2))

n=2
i=0
for neqn in range(3):
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    rms_norm = np.zeros((nalpha*nbeta, nmc))
    def func_rms(imc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        
        line=[]
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line.append( (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta]) /
                                               delta_h[events])**n ))**(1/n))
                                              
        return line
    
    out_rms = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc)
        for imc in range(nmc)
    )
    
    rms_norm = np.array(out_rms)
    
    heatmap = np.sum(rms_norm[:,:], axis=0)/nmc
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    MIN1 = np.array(np.where(heatmap[:, :]==heatmap[:, :].min()), dtype=int)
    
    best_entrainment_wet[i] = [nalpha-1-MIN1[0][0], MIN1[1][0]]

    i+=1
    

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
                line.append( (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                                  + nn0*nT0 * imer
                                                                  + nT0 * in0
                                                                  + iT0, nbeta*ialpha+ibeta]) /
                                               delta_h[events])**n))**(1/n))
                
        return line
    
    out = Parallel(n_jobs=-1)(
        delayed(func_rms)(imc)
        for imc in range(nmc)
    )
    
    rms_norm = np.array(out)
    
    heatmap = np.sum(rms_norm[:,:], axis=0)/nmc
    heatmap = np.array([heatmap[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
    
    MIN1 = np.array(np.where(heatmap[:, :]==heatmap[:, :].min()), dtype=int)
    
    best_entrainment_wet[i] = [nalpha-1-MIN1[0][0], MIN1[1][0]]

    i+=1
    
error_wet = []

entrainment = np.array(best_entrainment_wet, dtype=int)
i = 0
n = 2
for neqn in range(3):
    
    eqn = ['dry', 'wet', 'ice'][neqn]
    Htop = [Htop_dry, Htop_wet, Htop_ice][neqn]
    
    ialpha, ibeta = entrainment[i]
    line=[]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]  
        
        line.append( (1/ngood * sum((abs(rand_h2[events,imc] - Htop[nmer*nn0*nT0*nalpha*nbeta * events
                                                                                           + nn0*nT0*nalpha*nbeta * imer
                                                                                           + nT0*nalpha*nbeta * in0
                                                                                           + nalpha*nbeta * iT0
                                                                                           + nbeta*ialpha + ibeta]) /
                                       delta_h[events])**n ))**(1/n))
    
    error_wet.append((round(np.quantile(line, 0.025),1), round(np.mean(line),1), round(np.quantile(line, 0.975),1)))
    i+=1


for la in range(2):
    
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    ialpha, ibeta = entrainment[i]
    line=[]
    
    for imc in range(nmc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]  
    
        line.append( (1/ngood * sum((abs(rand_h2[events,imc] - outH[nmer*nn0*nT0 * events 
                                                          + nn0*nT0 * imer
                                                          + nT0 * in0
                                                          + iT0, nbeta*ialpha+ibeta]) /
                                       delta_h[events])**n))**(1/n))
        
    error_wet.append((round(np.quantile(line, 0.025),1), round(np.mean(line),1), round(np.quantile(line, 0.975),1)))
    i+=1

for imc in range(nmc):
    in0 = n0_indexes[imc]
    iT0 = T0_indexes[imc]
    imer = mer_indexes[imc]  

    line.append( (1/ngood * sum((abs(rand_h2[events,imc] - a2[0]*MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                       + nn0*nT0*nalpha*nbeta * imer
                                                                                       + nT0*nalpha*nbeta * in0
                                                                                       + nalpha*nbeta * iT0 for ievent in events]]**a2[1]) /
                                   delta_h[events])**n))**(1/n))
    
error_wet.append((round(np.quantile(line, 0.025),1), round(np.mean(line),1), round(np.quantile(line, 0.975),1)))


# %% BOXPLOT

tuned_wet = False

if tuned_wet:
    entrainment = np.array(best_entrainment_wet, dtype=int)
    a1, b1 = a2[0], a2[1]
    title = f'Models tuned over {ngood} wet eruptions'
    maxy = 19.4
    miny = -6.3
    errors = error_wet
    
else:
    entrainment = np.array(best_entrainment)
    a1, b1 = linfit 
    title = 'Models tuned over whole database'
    maxy = 20.8
    miny=-6.3
    errors = error_all


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

with plt.style.context(["science"]): 
    
    # fig = plt.figure(figsize=(15,8), layout='constrained')
    fig=plt.figure(figsize=(14,9), layout='constrained')
    ax = plt.subplot(111)
    # plt.title(title, fontsize=16)

    meanprops0 = dict(markeredgecolor='black', markerfacecolor='None', marker='s', markersize=1.0)
    whiskerprops0 = dict(linestyle='-', color='#ff006e', lw=0.8)
    boxprops0 = dict(linestyle='-', color='#ff006e', lw=1)
    # boxprops0 = dict(linestyle='-', color='grey', lw=1.2, facecolor='lightsteelblue', alpha=0.6)
    medianprops0 = dict(linestyle='-', color='#ff006e', lw=1)

    meanprops1 = dict(markeredgecolor='black', markerfacecolor='none', marker='s', markersize=1.0)
    whiskerprops1 = dict(linestyle='-', color='firebrick', lw=0.8)
    boxprops1 = dict(linestyle='-', color='firebrick', lw=1)
    medianprops1 = dict(linestyle='-', color='firebrick', lw=1)
    
    meanprops2 = dict(markeredgecolor='black', markerfacecolor='none', marker='s', markersize=1.0)
    whiskerprops2 = dict(linestyle='-', color='#e76f51', lw=0.8)
    boxprops2 = dict(linestyle='-', color='#e76f51', lw=1)
    medianprops2 = dict(linestyle='-', color='#e76f51', lw=1)
    
    meanprops3 = dict(markeredgecolor='black', markerfacecolor='none', marker='s', markersize=1.0)
    whiskerprops3 = dict(linestyle='-', color='#2a9d8f', lw=0.8)
    boxprops3 = dict(linestyle='-', color='#2a9d8f', lw=1)
    medianprops3 = dict(linestyle='-', color='#2a9d8f', lw=1)
    
    meanprops4 = dict(markeredgecolor='black', markerfacecolor='none', marker='s', markersize=1.0)
    whiskerprops4 = dict(linestyle='-', color='#457b9d', lw=0.8)
    boxprops4 = dict(linestyle='-', color='#457b9d', lw=1)
    medianprops4 = dict(linestyle='-', color='#457b9d', lw=1)
    
    meanprops5 = dict(markeredgecolor='black', markerfacecolor='none', marker='s', markersize=1.0)
    whiskerprops5 = dict(linestyle='-', color='k', lw=0.8)
    boxprops5 = dict(linestyle='-', color='k', lw=1)
    medianprops5 = dict(linestyle='-', color='k', lw=1)
    
    absc_under = np.linspace(0,100, len(event_ind))
    ratio_miny = miny/maxy
    plt.ylim(miny, maxy)
    plt.xlim(-5, 102.5)
    plt.yticks([0, 5, 10, 15, 20], minor=False, fontsize=13)
    plt.yticks(range(int(maxy)), minor=True)
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.grid(which='minor', axis='y', linestyle='-', linewidth=0.3, alpha=0.5)
    plt.tick_params(right=True, left=True, axis='y', color='k', length=3.5, width=1, labelright=True, labelsize=13)
    plt.tick_params(bottom=False, top=False, axis='x')

    plt.text(-absc_under[1], miny+0.3, r'Eruption (MER in $kg/s$)',
            horizontalalignment='center',
            verticalalignment='bottom',
            rotation='vertical',
            fontsize=13)

    width = 100//len(events)
    k=0
    for ievent in events:
        merlist = MER[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                + nn0*nT0*nalpha*nbeta * mer_indexes[imc]
                                                + nT0*nalpha*nbeta * n0_indexes[imc]
                                                + nalpha*nbeta * T0_indexes[imc] for imc in range(nmc)]]
        if k%2 == 0:
            x = plt.bar(absc_under[k], eru_d_h[ievent], bottom=eru_h_avl[ievent]-eru_d_h[ievent]/2, width=width, align='center', color='k' , alpha=0.3)
            plt.bar(absc_under[k], -11, bottom=0, width=width, align='center', color='k' , alpha=0.09)
            plt.bar(absc_under[k], 55, bottom=0, width=width, align='center', color='k' , alpha=0.09)
            
        else:
            x = plt.bar(absc_under[k], eru_d_h[ievent], bottom=eru_h_avl[ievent]-eru_d_h[ievent]/2, width=width, align='center', color='k' , alpha=0.2, label='IVESPA (estimated 95\% range)')
            plt.bar(absc_under[k], -11, bottom=0, width=width, align='center', color='k' , alpha=0.03)
            plt.bar(absc_under[k], 55, bottom=0, width=width, align='center', color='k' , alpha=0.03)
                        
        plt.text(absc_under[k], -0.6, f'{ivespa_ID[ievent]} ({round(mmer[ievent]/(10**int(np.log10(mmer[ievent]))),1)} e{int(np.log10(mmer[ievent]))})',
                   horizontalalignment='center',
                   verticalalignment='top',
                   rotation='vertical',
                   fontsize=13)
        
        widths = 0.47
        positions = [absc_under[k]-1.4, absc_under[k]-0.84, absc_under[k]-0.28, absc_under[k]+0.28, absc_under[k]+0.84, absc_under[k]+1.4]
        
        b = plt.boxplot(Htop_dry[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                               + nn0*nT0*nalpha*nbeta * mer_indexes[i]
                                               + nT0*nalpha*nbeta * n0_indexes[i]
                                               + nalpha*nbeta * T0_indexes[i]
                                               + nbeta*entrainment[0,0] + entrainment[0,1] for i in range(nmc)]].ravel(), positions=[positions[0]], boxprops=boxprops0, showmeans=False, meanprops=meanprops0, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops0, capwidths=0.5, widths=widths, capprops=whiskerprops0, medianprops=medianprops0)
        
        c = plt.boxplot(Htop_wet[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                               + nn0*nT0*nalpha*nbeta * mer_indexes[i]
                                               + nT0*nalpha*nbeta * n0_indexes[i]
                                               + nalpha*nbeta * T0_indexes[i]
                                               + nbeta*entrainment[1,0] + entrainment[1,1] for i in range(nmc)]].ravel(), positions=[positions[1]], boxprops=boxprops1, showmeans=False, meanprops=meanprops1, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops1, capwidths=0.5, widths=widths, capprops=whiskerprops1, medianprops=medianprops1)
        
        d = plt.boxplot(Htop_ice[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                               + nn0*nT0*nalpha*nbeta * mer_indexes[i]
                                               + nT0*nalpha*nbeta * n0_indexes[i]
                                               + nalpha*nbeta * T0_indexes[i]
                                               + nbeta*entrainment[2,0] + entrainment[2,1] for i in range(nmc)]].ravel(), positions=[positions[2]], boxprops=boxprops2, showmeans=False, meanprops=meanprops2, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops2, capwidths=0.5, widths=widths, capprops=whiskerprops2, medianprops=medianprops2)
        
        e = plt.boxplot(out_Htop0[nmer*nn0*nT0 * ievent : nmer*nn0*nT0 * (ievent+1), nbeta*entrainment[3,0] + entrainment[3,1]].ravel(), positions=[positions[3]], boxprops=boxprops3, showmeans=False, meanprops=meanprops3, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops3, capwidths=0.5, widths=widths, capprops=whiskerprops3, medianprops=medianprops3)
        
        f = plt.boxplot(out_Htop2[nmer*nn0*nT0 * ievent : nmer*nn0*nT0 * (ievent+1), nbeta*entrainment[4,0] + entrainment[4,1]].ravel(), positions=[positions[4]], boxprops=boxprops4, showmeans=False, meanprops=meanprops4, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops4, capwidths=0.5, widths=widths, capprops=whiskerprops4, medianprops=medianprops4)
        
        g = plt.boxplot(a1*merlist**b1, positions=[positions[5]], boxprops=boxprops5, showmeans=False, meanprops=meanprops5, 
                        showfliers=False, whis=(2.5, 97.5), whiskerprops=whiskerprops5, capwidths=0.5, widths=widths, capprops=whiskerprops5, medianprops=medianprops5)
        
        k+=1
    
    ax.legend(handles=[x, b['whiskers'][0], c['boxes'][0], d['medians'][0], e['medians'][0], f['medians'][0], g['medians'][0]],
              labels=['IVESPA (estimated 95\% range)', rf'dry Plumerise, error={errors[0][1]} ({errors[0][0]}-{errors[0][2]})', 
                      rf'wet Plumerise, error ={errors[1][1]} ({errors[1][0]}-{errors[1][2]})', rf'ice Plumerise, error={errors[2][1]} ({errors[2][0]}-{errors[2][2]})',
                      rf'Geneva without condensation, error={errors[3][1]} ({errors[3][0]}-{errors[3][2]})', rf'Geneva with condensation, error={errors[4][1]} ({errors[4][0]}-{errors[4][2]})',
                      rf'Empirical power-law, error={errors[5][1]} ({errors[5][0]}-{errors[5][2]})'], 
                      fontsize=16, loc='upper left', frameon=True, framealpha=0.4, ncols=2, handletextpad=0.4, 
                      borderpad=0.3, handlelength=1.1, columnspacing=1.0, markerscale=3.0)

    plt.xticks([])
    plt.ylabel(r'Plume top height $(\overline{H_{top}})$\;[km \;a.v.l.]', fontsize=18)
    # plt.savefig(f'{dir_fig}/Figure3.pdf', dpi=300, format='pdf')
    # plt.close()

# %%

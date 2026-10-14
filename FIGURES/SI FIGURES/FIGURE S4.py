#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:08:40 2025

@author: titouanrenaud
"""

from scipy.stats import qmc
import os
import numpy as np
import pandas as pd
import plumerise as pr
from joblib import Parallel, delayed
import pickle
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scienceplots
import random
from matplotlib import rc
from scipy.optimize import curve_fit
from utils import load_input, getIVESPAtable

init_dir = load_input() #Drive data folder

outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs'

nevent = 130
nn0, nT0 = 5, 6
nmer = 11
nalpha, nbeta = 6, 10
neqn = 2

MC_n0_T0 = True
u0_uniform = True

eqn='dry'
outputs_file = os.path.join(outputs_dir, f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs_dry = pd.read_csv(outputs_file, low_memory = False)
Htop = np.array(outputs_dry['Htop'])
MER = np.array(outputs_dry['input_mer'])

n0list = list(set(list(outputs_dry['input_n0'])))
n0list.sort()

T0list = list(set(list(outputs_dry['input_T0'])))
T0list.sort()

alphalist = list(set(list(outputs_dry['alpha'])))
alphalist.sort()

betalist = list(set(list(outputs_dry['beta'])))
betalist.sort()

ivespa = getIVESPAtable()


ivespa_ID = list(outputs_dry.ivespa_ID[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                for ievent in range(nevent)]])

ivespa_indexes = [list(ivespa['Ivespa_ID']).index(event) for event in ivespa_ID]

eru_h = np.array(ivespa['TopHeight'][ivespa_indexes])                   # Plume top height [km a.s.l]
eru_d_h = np.array(ivespa['TopHeight_Uncertainty'][ivespa_indexes])     # Plume top height uncertainty
eru_vh = np.array(ivespa['VentAltitude'][ivespa_indexes])/1000          # Vent altitude [m a.s.l]
eru_h_avl = eru_h - eru_vh

# %% DISTRIBUTIONS

ievent=44

nmc = 10000
temps0=time.time()
        
rand_h2 = np.zeros(nmc)

for i in range(nmc):
    while True:
        rand_h2[i] = random.gauss(eru_h_avl[ievent], eru_d_h[ievent]/2)
        if abs(rand_h2[i]-eru_h_avl[ievent]) < 2*eru_d_h[ievent] and rand_h2[i] > 0: # get rid of extreme values
            break


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
  
ialpha, ibeta = 2,1
Hmod = Htop[[nmer*nn0*nT0*nalpha*nbeta * ievent
                    + nn0*nT0*nalpha*nbeta * mer_indexes[imc]
                    + nT0*nalpha*nbeta * n0_indexes[imc]
                    + nalpha*nbeta * T0_indexes[imc]
                    + nbeta*ialpha + ibeta for imc in range(nmc)]]

# %% FIGURE

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

labels = ['IVESPA best-estimate']
x1 = [ eru_h_avl[ievent]]
x2 = [3.0, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]

absc = np.linspace(2.5, 7.3)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

with plt.style.context(["science", 'grid']):
    plt.figure(figsize=(10,3.5), layout='constrained')
    plt.xlabel(r'Top height $H_{top}$ [km above vent level]', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xlim(2.7, 7.1)
    plt.ylim(0, 620)

    plt.xticks(x2,labels=x2, minor=True, fontsize=14)
    
    hist1 = plt.hist(Hmod, bins=np.linspace(min(Hmod), max(Hmod), 50), label='Predicted heights Hmod', color='#457b9d', alpha=0.6)
    a1, b1 = curve_fit(gaus, (hist1[1][1:]+hist1[1][:-1])/2, hist1[0])
    
    hist2 = plt.hist(rand_h2, bins=np.linspace(min(rand_h2), max(rand_h2), 50), label='Observed heights Hobs', color='firebrick', alpha=0.6 )
    a2, b2 = curve_fit(gaus, (hist2[1][1:]+hist2[1][:-1])/2, hist2[0])
    
    plt.xticks(x1, labels=labels, minor=False, fontsize=13)
       
    plt.legend(fontsize=14)
    plt.savefig('FIGURES4_distri_heights.pdf', bbox_inches='tight', dpi=300, format='pdf')
    plt.show()     
# %%

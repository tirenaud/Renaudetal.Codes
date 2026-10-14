#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 2026

@author: titouanrenaud
"""

from importlib.metadata import distribution
import os
import numpy as np
import pandas as pd
import plumerise as pr
import random
import time
import matplotlib.pyplot as plt
from utils import load_input

init_dir = load_input() #Drive data folder

dir_fig = os.getcwd()

outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs/outputs_130_nn0=5_nT0=6_nmer=11_nalpha=6_nbeta=10_plumerise_dry_acheve'
outputs_file = pd.read_csv(outputs_dir, low_memory = False)

outputs_interest = outputs_file[outputs_file['ivespa_ID'] == 'EYJ2010_01']

inputs_mer = outputs_interest['input_mer'].values
inputs_mer = sorted(list(set(inputs_mer)))  # Get unique values

inputs_n0 = outputs_interest['input_n0'].values
inputs_n0 = sorted(list(set(inputs_n0)))  # Get unique values

inputs_T0 = outputs_interest['input_T0'].values
inputs_T0 = sorted(list(set(inputs_T0)))  # Get unique values

nmc = 10000

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

mer_indexes = np.sort(mer_indexes)
lim = np.where(mer_indexes == 5)[0][0]

array_mer = outputs_interest['input_mer'].values


# %% FIGURE with several distributions: n0, T0 then MER, u0 and r0

plt.figure(figsize=(10, 9), tight_layout=True)

### distribution of n0
ax = plt.subplot(323)
bins = [0.0025, 0.0175, 0.0325, 0.0475, 0.0625, 0.0775]
labels = [round(n0*100,1) for n0 in inputs_n0]
x1 = inputs_n0
ax.set_xticks(x1, labels=labels, minor=False, fontsize=14)

ax.hist(np.array(inputs_n0)[n0_indexes], density=False, bins=bins, rwidth=0.7, 
             weights=np.ones(nmc)/nmc, align='mid', color='#457b9d')
ax.set_title(r'Gas mass fraction $n0\;[w.t.\%]$', fontsize=14)
ax.set_ylabel(r'Probability', fontsize=14)

### distribution of T0
ax = plt.subplot(324)
bins = [840.0, 960.0, 1080.0, 1200.0, 1320.0, 1440.0, 1560.0]
labels = [f'{T0:.0f}' for T0 in inputs_T0]
x1 = inputs_T0
ax.set_xticks(x1, labels=labels, minor=False, fontsize=14)

ax.hist(np.array(inputs_T0)[T0_indexes], bins=bins, rwidth=0.7, 
        weights=np.ones(nmc)/nmc, align='mid', color='#457b9d')
ax.set_title(r'Plume temperature $T0\;[K]$', fontsize=14)
ax.set_yticks([])

### distribution of MER
ax = plt.subplot(311)
# ax.set_xscale('log')

width1 = inputs_mer[1] - inputs_mer[0]  # Calculate the width of the bins
width2 = inputs_mer[7] - inputs_mer[6]  # Calculate the width of the bins
bins1 = [inputs_mer[0] - width1 / 2] + [x + width1 / 2 for x in inputs_mer[:6]]
bins2 = [x + width2 / 2 for x in inputs_mer[5:]]
width = width1 * 0.8

mer_indexes = np.sort(mer_indexes)
lim = np.where(mer_indexes == 6)[0][0]
array_mer = outputs_interest['input_mer'].values

h1 = ax.hist(array_mer[[nn0*nT0*nalpha*nbeta * imer for imer in mer_indexes[:lim]]], weights=np.ones(lim)/nmc, bins=bins1, align='mid', color='#457b9d', rwidth=0.7)
h2 = ax.hist(array_mer[[nn0*nT0*nalpha*nbeta * imer for imer in mer_indexes[lim:]]], weights=np.ones(nmc-lim)/nmc, bins=bins2, align='mid', color='#457b9d', rwidth=0.7*width1/width2)
ax.set_title(r'Mass Eruption Rate $\overline{MER}\;[kg/s]$', fontsize=14)
ax.set_ylabel(r'Probability', fontsize=14)
x1 = [inputs_mer[0], inputs_mer[5], inputs_mer[-1]]
labels = [r'$3.38e5$', r'$5.29e5$', r'$8.10e5$']
x2 = inputs_mer
ax.set_xticks(x1, labels=labels, minor=False, fontsize=14)
ax.set_xticks(x2, minor=True, labels=[])

### distribution of r0
ax = plt.subplot(325)
array_r0 = outputs_interest['input_r0'].values
h3 = ax.hist(array_r0[[nn0*nT0*nalpha*nbeta * mer_indexes[k]
                   + nT0*nalpha*nbeta * n0_indexes[k]
                   + nalpha*nbeta + T0_indexes[k] for k in range(nmc)]], weights=np.ones(nmc)/nmc, bins=15, color='#457b9d', rwidth=1)
ax.set_ylabel(r'Probability', fontsize=14)
ax.set_title(r'Plume radius $r0\;[m]$', fontsize=14)
ax.tick_params(axis='x', labelsize=14)

### distribution of u0
ax = plt.subplot(326)
array_u0 = outputs_interest['input_u0'].values
ax.hist(array_u0[[nn0*nT0*nalpha*nbeta * mer_indexes[k]
                   + nT0*nalpha*nbeta * n0_indexes[k]
                   + nalpha*nbeta + T0_indexes[k] for k in range(nmc)]]/3.6, weights=np.ones(nmc)/nmc, bins=10, color='#457b9d', rwidth=1, align='mid')
ax.set_title(r'Plume initial vertical velocity $u0\;[m/s]$', fontsize=14)
ax.set_yticks([])
ax.tick_params(axis='x', labelsize=14)

plt.savefig('FIGURES3_distri_inputs.pdf', bbox_inches='tight', dpi=300, format='pdf')
plt.show()

# %%

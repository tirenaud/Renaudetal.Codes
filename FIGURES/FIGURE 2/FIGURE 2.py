#%% imports 

import os
from tkinter import font
import numpy as np
import pandas as pd
import plumerise as pr
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import scienceplots
from utils import plotly_to_cmap, load_input
import textwrap

init_dir = load_input() #Drive data folder

dir_fig = os.getcwd()  #directory to save figures

# nputs for nevent=130, nn0=5, nT0=6, nmer=11
inputs_dir = f'{init_dir}/MONTE CARLO RUNS/inputs/inputs_nevent=130_nn0=5_nT0=6_nmer=11'

# outputs, have to select matlab or plumerise version later
outputs_dir = f'{init_dir}/MONTE CARLO RUNS/outputs'

nevent = 130
nn0, nT0 = 5, 6
nmer = 11
nalpha, nbeta = 6, 10

eqn='dry' #uploading dry Plumerise's outputs
outputs_file = os.path.join(outputs_dir, 
                            f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs_dry = pd.read_csv(outputs_file, low_memory = False)
Htop_dry = np.array(outputs_dry['Htop'])
MER = np.array(outputs_dry['input_mer'])

eqn = 'wet' #uploading wet Plumerise's outputs
outputs_file = os.path.join(outputs_dir, 
                            f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
outputs = pd.read_csv(outputs_file, low_memory = False)
Htop_wet = np.array(outputs['Htop'])

eqn = 'ice' #uploading ice Plumerise's outputs
outputs_file = os.path.join(outputs_dir, 
                            f'outputs_{nevent}_nn0={nn0}_nT0={nT0}_nmer={nmer}_nalpha={nalpha}_nbeta={nbeta}_plumerise_{eqn}_acheve')
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

# %% Creating legend

plt.figure(layout='constrained', figsize=(18,16))
l1, = plt.plot(1, 1, "or", markersize=20)
l2, = plt.plot([0,1], [0,1], 'k-')

custom_handle = Line2D(
    [], [], 
    lw=0)

plt.text(1, 0.5, r'\textbf{\%}', ha='center', va='center', fontsize=35, color='w',
                                 path_effects=[pe.withStroke(linewidth=1, foreground="k")])

label1 = textwrap.fill(
    "Best entrainment parameters pair",
    width=18
)

label2 = textwrap.fill(
    "Probability to outperform the best entrainment pair (only shown if $>$ 5\%)",
    width=18
)

label3 = textwrap.fill(
    "3D models and lab experiments contraints",
    width=18
)

plt.legend(
    [l1, custom_handle, custom_handle],
    [label1, label2, label3],
    loc='upper left',
    bbox_to_anchor=(0.02, 0.5, 0.28, 0.4),
    bbox_transform=plt.gca().transAxes,
    frameon=True,
    fontsize=25,
    labelspacing = 0.8
)

plt.xlim(0,3)
plt.ylim(0,3)

plt.savefig(f'{dir_fig}/legend.pdf', format='pdf')
plt.close()

# %% PLOTS

events = np.array(range(nevent)) #defining the list of events taken. Here all 130 events, no selection.
ngood = nevent #number of events in the selection. Here 130

best_ab = np.zeros((5,2)) #list that will keep the best (alpha, beta) defined over all the events. Named alpha* and beta*

n = 2 # used in definition of RMS. 

with open(f'{dir_fig}/ALL_EVENTS.txt', "w") as fichier:
    fichier.write(f"ALL EVENTS, Nevents={ngood}\n\n")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

### For the 3 Plumerise models: dry, wet & ice
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
                line.append( (1/ngood * sum((abs(rand_h2[events,imc] - Htop[[nmer*nn0*nT0*nalpha*nbeta * ievent
                                                                                                   + nn0*nT0*nalpha*nbeta * imer
                                                                                                   + nT0*nalpha*nbeta * in0
                                                                                                   + nalpha*nbeta * iT0
                                                                                                   + nbeta*ialpha + ibeta for ievent in events]]) /
                                               delta_h[[ievent for ievent in events]])**n ))**(1/n))
                                              
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
    
    best_ab[i,:] = [alpha_star, beta_star]
    
    with plt.style.context(["science"]): 
        alphaticks = [round(alphalist[-i],4) for i in range(1,nalpha+1)]
        betaticks = [round(betalist[i],4) for i in range(nbeta)]
        
        fig = plt.figure(layout='constrained', figsize=(18,16))
        
        plt.plot(int(MIN1[1][0]), int(MIN1[0][0]), "or", markersize=20) #showing the entrainement parametrization minimizing the error
        
        if neqn==0:
            plt.xlabel(r'wind entrainment coefficient $\beta$', fontsize=40)
            plt.xticks(range(nbeta), betaticks, fontsize=35)
            
            plt.ylabel(r'radial entrainment coefficient $\alpha$', fontsize=40)
            plt.yticks(range(nalpha), alphaticks, fontsize=35)
        elif neqn==1:
            plt.ylabel(r'$\alpha$', fontsize=45)
            plt.yticks(range(nalpha), alphaticks, fontsize=45)

            plt.xlabel(None)
            plt.xticks([])
        elif neqn==2:
            plt.xlabel(None)
            plt.xticks([])

            plt.ylabel(None)
            plt.yticks([])

        for ialpha in range(nalpha-1):
            plt.plot([-0.5, 9.5], [nalpha-1.5-ialpha, nalpha-1.5-ialpha], color='k', lw=0.8)
            
        for ibeta in range(nbeta-1):
            plt.plot([ibeta+0.5, ibeta+0.5], [-0.5, 5.5], color='k', lw=0.8)
        
        ax = fig.gca()
        
        alphas = np.ones(np.shape(heatmap))
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                if (ialpha == alpha_star) and (ibeta == beta_star):
                    continue
                
                prop_under=0
                prop_over=0
                for s in range(nmc):
                    diff = rms_norm[s, nbeta*ialpha+ibeta] - rms_norm[s, nbeta*alpha_star+beta_star]
                    
                    if diff > 0:
                        prop_over+=1
                    else:
                        prop_under+=1
                        
                if prop_under/nmc < 0.05:                    
                    ax.add_patch(Rectangle((ibeta-0.475,nalpha-1.475-ialpha), 0.95, 0.95, linewidth=0, fill=None, hatch='///'))
                    
                else:
                    if neqn==0:
                        plt.text(ibeta, nalpha-1-ialpha, rf'\textbf{{{int(prop_under/nmc * 100)}\%}}', ha='center', va='center', fontsize=35, color='w',
                                 path_effects=[pe.withStroke(linewidth=1, foreground="k")])

                    else:
                        plt.text(ibeta, nalpha-1-ialpha, rf'\textbf{{{int(prop_under/nmc *100)}\%}}', ha='center', va='center', fontsize=45, color='w',
                                 path_effects=[pe.withStroke(linewidth=1, foreground="k")])

        plt.imshow(heatmap[:,:], cmap=plotly_to_cmap('Sunsetdark'), vmin=1.034, vmax=1.513)
        
        if neqn==0:
            cbar = plt.colorbar()
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(45)
    
            cbar.ax.set_ylabel(r'Error metric $\overline{E}_{p}$', fontsize=50)

        plt.savefig(dir_fig+f'/{eqn} Plumerise, ALL EVENTS, {ngood} events.pdf', format='pdf')
        
        with open(dir_fig+"/ALL_EVENTS.txt", "a") as fichier:
            fichier.write(f"{eqn} Plumerise, alpha*={alphalist[alpha_star]}, beta*={betalist[beta_star]}, err*={heatmap[MIN1][0]}\n\n")
            ### keeping (alpha*, beta*), minimizing the error. +err* = error for that entrainement parametrization

        plt.close()

        i += 1        

### Same for the two Geneva models. Lambda=0.0 & Lambda=1e-2
for la in range(2): 
    val = ['0.0', '1e-2'][la]
    outH = [out_Htop0, out_Htop2][la]
    title = ['without condensation, $\lambda=0.0$', 'with condensation, $\lambda=1e-2$'][la]
    
    rms_norm = np.zeros((nalpha*nbeta, nmc))
    def func_rms(imc):
        in0 = n0_indexes[imc]
        iT0 = T0_indexes[imc]
        imer = mer_indexes[imc]
        line=[]
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                line.append( (1/ngood * sum((abs(rand_h2[events,imc] - outH[[nmer*nn0*nT0 * ievent 
                                                                  + nn0*nT0 * imer
                                                                  + nT0 * in0
                                                                  + iT0 for ievent in events],nbeta*ialpha+ibeta]) /
                                               delta_h[[ievent for ievent in events]])**n))**(1/n))
                
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
    
    best_ab[i,:] = [alpha_star, beta_star]
    
    with plt.style.context(["science"]): 
        heat1 = np.sum(rms_norm[:,:], axis=0)/nmc
        heat1 = np.array([heat1[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)])
        
        alphaticks = [round(alphalist[-i],4) for i in range(1,nalpha+1)]
        betaticks = [round(betalist[i],4) for i in range(nbeta)]
        
        MIN1 = np.where(heat1[:, :]==heat1[:, :].min())
        
        heat2 = np.zeros(nalpha*nbeta)
        
        for s in range(nmc):
            index = np.where(rms_norm[s,:] == min(rms_norm[s,:]))
            heat2[int(index[0][0])] += 1
        
        heat2 = np.array([heat2[nbeta*ialpha:nbeta*(ialpha+1)] for ialpha in range(nalpha-1, -1, -1)]) 
        
        prop_star = round(heat2[:,:].max()*100/nmc,1)
        
        fig = plt.figure(layout='constrained', figsize=(18,16))
        
        
        plt.plot(int(MIN1[1][0]), int(MIN1[0][0]), "or", markersize=20) #showing the entrainement parametrization minimizing the error

        plt.xlabel(r'$\beta$', fontsize=30)
        plt.xticks(range(nbeta), betaticks, fontsize=25)
        
        plt.ylabel(r'$\alpha$', fontsize=30)
        plt.yticks(range(nalpha), alphaticks, fontsize=25)

        if la==0:
            plt.xlabel(r'$\beta$', fontsize=45)
            plt.xticks(range(nbeta), betaticks, fontsize=45)

            plt.ylabel(r'$\alpha$', fontsize=45)
            plt.yticks(range(nalpha), alphaticks, fontsize=45)
        elif la==1:
            plt.xlabel(r'$\beta$', fontsize=45)
            plt.xticks(range(nbeta), betaticks, fontsize=45)

            plt.ylabel(None)
            plt.yticks([])
       
        ax = fig.gca()
        
        for ialpha in range(nalpha-1):
            plt.plot([-0.5, 9.5], [nalpha-1.5-ialpha, nalpha-1.5-ialpha], color='k', lw=0.8)
            
        for ibeta in range(nbeta-1):
            plt.plot([ibeta+0.5, ibeta+0.5], [-0.5, 5.5], color='k', lw=0.8)
        
        for ialpha in range(nalpha):
            for ibeta in range(nbeta):
                if (ialpha == alpha_star) and (ibeta == beta_star):
                    continue
                
                prop_under=0
                prop_over=0
                for s in range(nmc):
                    diff = rms_norm[s, nbeta*ialpha+ibeta] - rms_norm[s, nbeta*alpha_star+beta_star]
                    
                    if diff > 0:
                        prop_over+=1
                    else:
                        prop_under+=1
                    
                if prop_under/nmc < 0.05:
                    ax.add_patch(Rectangle((ibeta-0.475,nalpha-1.475-ialpha), 0.95, 0.95, linewidth=0, fill=None, hatch='///'))
                      
                else:
                    plt.text(ibeta, nalpha-1-ialpha, rf'\textbf{{{int(prop_under/nmc *100)}\%}}', ha='center', va='center', fontsize=45, color='w',
                                 path_effects=[pe.withStroke(linewidth=1, foreground="k")])
                    
        plt.imshow(heatmap[:,:], alpha=alphas, cmap=plotly_to_cmap('Sunsetdark'), vmin=1.034, vmax=1.513)
        
        # plt.savefig(dir_fig+f'/Geneva lambda= {val}, ALL EVENTS, {ngood} events.pdf', format='pdf')
        # plt.close()

        with open(dir_fig+"/ALL_EVENTS.txt", "a") as fichier:
            fichier.write(f"Geneva lambda= {val}, alpha*={alphalist[alpha_star]}, beta*={betalist[beta_star]}, err*={heatmap[MIN1][0]}\n\n")

        i += 1  

# %%

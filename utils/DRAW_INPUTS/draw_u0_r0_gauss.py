#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:18:09 2025

@author: titouanrenaud
"""

import plumerise as pr
import numpy as np

def draw_u0_r0_gauss(mer, rho_B0, vh0, T0, n0, atm):
    """
    draw_u0_r0_gauss draws the vent radius and exit velocity that match the mass eruption
    rate without plume collapse

    Inputs :  mer          -> MER [kg/s]
              rho_B0       -> Bulk density of the plume at vent height [kg/m^3]
              vh0          -> Vent altitude [m]
              T0           -> Exit temperature [K]
              n0           -> Total water mass fraction [-]
              atm          -> Atmospheric profil (x6 double array)

    Outputs : r0           -> Drawn vent radius [m]
              u0           -> Drawn exit velocity [m/s]
              success      -> unable to find r0 and u0 matching the MER [bool]
                              (can happen with extremely low MER)
                              
    """    
    
    Cs = 1100
    Ca = 1000
    Cv = 1810
    Ra = 287
    
    alpha, beta = 0.025, 0.1
    eqn = 'dry'

    u0 = max((np.log(mer) - 1.08)/0.0234, 7)
    r0 = np.sqrt(mer/( np.pi * u0 * rho_B0 ))
    
    Params = pr.set_params(ks=alpha, kw=beta, gas_thrust_entrainment = False,
                           Cs=Cs, Ca=Ca, Cv=Cv, Ra=Ra)

    SolParams = pr.set_solver_params(eqn = eqn, OutStep=-1)

    IC = pr.set_IC(L0=r0, U0=u0, n0=n0, T0=T0, z0=vh0)

    soln,plume_state,flag, MetVals = pr.solve(Params, SolParams, IC, atm, verbose=False)
    
    if not plume_state['Collapse']:
        return (np.float64(r0), np.float64(u0), True)
    
    return (np.nan, np.nan, False)
    
        
        
            
        
        
        
    
    
    
    
    
    
    
    
    


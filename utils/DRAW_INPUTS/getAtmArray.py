#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:53:14 2025

@author: titouanrenaud
"""

import numpy as np

def getAtmArray(atmmatrix):
    """
    This function reads the averaged atmospheric profiles from ERA database and returns an array used in the models.
    
    Input: atmospheric matrix file
    
    Output: array with:
        * Altitudes
        * Temperatures
        * Pressures
        * Wind in (x, y) space
        * Relative humidity

    """

    atmmatrix[np.isnan(atmmatrix)] = 0

    atmo = np.zeros(atmmatrix.shape)

    atmo[:, 0] = atmmatrix['Altitude (m a.s.l)']
    atmo[:, 1] = atmmatrix['Temperature (K)']
    atmo[:, 2] = atmmatrix['Pressure (Pa)']
    atmo[:, 4] , atmo[:, 3]  = np.sqrt(atmmatrix['Zonal wind speed (m/s)']**2+ atmmatrix['Meridional wind speed (m/s)']**2) , np.degrees(np.arccos(atmmatrix['Meridional wind speed (m/s)']/np.sqrt(atmmatrix['Zonal wind speed (m/s)']**2+ atmmatrix['Meridional wind speed (m/s)']**2)))
    atmo[:, 5] = atmmatrix['Relative humidity (%)']/100

    return atmo
"""This creates a polar graph of the flight trajectory"""   

import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
#import pandas as pd 
import matplotlib.pyplot as plt

"""Graph parameters"""
# These commands modify default figure properties to make them more legible
plt.rcParams['axes.linewidth']    = 2.5
plt.rcParams['lines.linewidth']   = 2.5
plt.rcParams['font.sans-serif']   = 'Arial'
plt.rcParams['font.weight']       = 'bold'
plt.rcParams['axes.labelweight']  = 'bold'
plt.rcParams['axes.titleweight']  = 'bold'
plt.rcParams['axes.labelsize']    = 16
plt.rcParams['axes.titlesize']    = 16
plt.rcParams['xtick.labelsize']   = 14
plt.rcParams['ytick.labelsize']  =  14
#plt.rcParams['axes.grid']         = True
plt.rcParams['xtick.major.size']  = 5
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.size']  = 5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['lines.markersize']  = 12
figdir = '/Users/elanacope/Documents/penn_state_reu/Figures/Graphs/'

"""Reading in data"""
#assigning the dataset
dataset = 'data_sample_d4.nc'
f = Dataset(dataset, mode='r')
tempgrp = f.groups['Radar_data']   


#This opens the netCDF file and assigns the variables that we want.
R = np.array(tempgrp.variables['Range'])
Theta = ((np.array(tempgrp.variables['Azimuth']))*(np.pi/180))

"""Plotting"""
#creating polar plot
ax=plt.subplot(111, polar=True)
ax.plot(Theta, R,'o-', label='Path 4 from (d)', ms=5, color='blue', markerfacecolor='gray') 
ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
ax.set_rgrids(radii=[0, 100, 200, 300, 400, 500], angle=112)
plt.savefig(figdir+'Polar_test_ncgenerator.png', dpi=900) #This saves the first page of figures
plt.show()

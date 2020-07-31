"""This creates a polar histogram of the azimuthal angles."""
import netCDF4
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
plt.rcParams['xtick.labelsize']   = 16
plt.rcParams['ytick.labelsize']  =  16
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

bins_number = 16  # the [0, 2pi) interval will be subdivided into this
bins = np.linspace(-np.pi, np.pi, bins_number + 1)

n, _, _ = plt.hist(Theta, bins) # Create histogram

"""Plotting"""
plt.clf()
width = 2 * np.pi / bins_number
ax = plt.subplot(1, 1, 1, projection='polar')
ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
ax.set_rgrids(radii=[0, 15, 30, 45, 60 ], angle=112)
bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, color='mediumslateblue', edgecolor='black', label="Number of flights in each direction")
for bar in bars:
    bar.set_alpha(0.5)
plt.legend(bbox_to_anchor = [0.4, 0.01])
plt.savefig(figdir+'PolarHistogram.png', dpi=900) #This saves the first page of figures
plt.show()

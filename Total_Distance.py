
"""This code creates a graph of total distance vs time."""

import netCDF4
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
#import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy import spatial
from matplotlib import rcParams

"""Graph Parameters"""
rcParams.update({'figure.autolayout': True}) #keeps the axis labels from getting cut off


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

"""Reading in Data"""
#assigning the dataset
dataset = 'data_sample_d4.nc'
f = Dataset(dataset, mode='r')
tempgrp = f.groups['Radar_data']   


#This opens the netCDF file and assigns the variables that come from the file 
# (polar coordinates and a latitude/longitude point)
R = np.array(tempgrp.variables['Range'])
Theta = ((np.array(tempgrp.variables['Azimuth']))*(np.pi/180))
lat = np.array(tempgrp.variables['Latitude'])
lon = np.array(tempgrp.variables['Longitude'])
time = np.array(tempgrp.variables['Time'])


"""Making calculations"""
#This converts from polar coordinates to cartesian coordinates.
n = R.size 
x=np.empty(n, )
y=np.empty(n, )
for i in range(n):
    x[i]= ( math.cos(Theta[i]) * R[i] )
    y[i]= ( math.sin(Theta[i]) * R[i] )

#Calculating step widths
dist = np.empty((n-1, ))  # pre-allocate space for the step lengths
for i in range(0,n-1):
    # we can use numpy's sqrt fn for numpy arrays
    dist[i] = np.sqrt( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )

total_dist = []
total = 0
for i in dist:
  total += i
  total_dist.append(total)

#make time same length
j = np.empty(n-1, )
for i in range(0, n-1):
    j[i] = 1
time_resized = np.empty(n-1, )
for i in range(0, n-1):
    time_resized[i] = time[i] * j[i]

"""Plotting graph"""
#plots the data and labels it

plt.plot(time_resized, total_dist, 'o-', label='Path 4 from (d)', ms=5, color='black', markerfacecolor='plum') 

#these add the labels, title, legend, etc. 
plt.xlabel('Time (s)')
plt.ylabel('Total Distance Traveled (m)')
plt.legend()
plt.savefig(figdir+'tser_tot.png', dpi=900) #This saves the first page of figures
plt.show()

print(time_resized.size)


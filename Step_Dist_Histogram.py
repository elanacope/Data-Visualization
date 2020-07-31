"""This code makes a histogram of step distances."""
import netCDF4
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
#import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy import spatial
from matplotlib import rcParams

"""Graph parameters"""
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

"""Reading in data"""
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
    
    # scipy.spatial can do this formula for us
    dist2 = spatial.distance.euclidean((x[i], y[i]), (x[i+1], y[i+1]))
    
    # check that they are the same
    assert np.isclose(dist[i], dist2)

"""Plotting the data"""
#making histogram
plt.hist(dist,16,range=(0,40), histtype='bar',
align='mid', color='lightsteelblue', label='Step Distances',edgecolor='black')
plt.title('Step Distances')
plt.xlabel('Distance traveled in 3s intervals (m)')
plt.ylabel('Number of occurrances')
plt.savefig(figdir+'Histogram.png', dpi=900) #This saves the first page of figures
plt.show()


"""This code creates a polar histogram of step angles."""
import netCDF4
import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
#import pandas as pd 
import matplotlib.pyplot as plt
import math
from scipy import spatial
from matplotlib import rcParams

"""Graph paramters"""
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
plt.rcParams['ytick.labelsize']  =  12
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


#This converts from polar coordinates to cartesian coordinates.
n = R.size 
x=np.empty(n, )
y=np.empty(n, )
for i in range(n):
    x[i]= ( math.cos(Theta[i]) * R[i] )
    y[i]= ( math.sin(Theta[i]) * R[i] )


"""Making calculations"""
#Calculating step widths
dist = np.empty((n-1, ))  # pre-allocate space for the step lengths
for i in range(0,n-1):
    # we can use numpy's sqrt fn for numpy arrays
    dist[i] = np.sqrt( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )
    
    # scipy.spatial can do this formula for us
    dist2 = spatial.distance.euclidean((x[i], y[i]), (x[i+1], y[i+1]))
    
    # check that they are the same
    assert np.isclose(dist[i], dist2)

#calculating distances between x and y
dx = np.empty(n-1, )
dy = np.empty(n-1, )
for i in range(0, n-1):
    dx[i]= x[i+1] - x[i]
    dy[i]= y[i+1] - y[i]
#angles between points
angles= np.empty(n-1, )
for i in range(0, n-1):
    angles[i] = math.atan2(dy[i], dx[i])

"""Plotting"""
bins_number = 16  # the [0, 2pi) interval will be subdivided into this
bins = np.linspace(-np.pi, np.pi, bins_number + 1)

n, _, _ = plt.hist(angles, bins) # Create frequency distribution

relfreq = n/angles.size #calculate relative frequency of each angle 

# Making polar histogram
plt.clf()
width = 2 * np.pi / bins_number
ax = plt.subplot(1, 1, 1, projection='polar')
ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', '']) #sets cardinal directions
ax.set_ylim([0, 0.12])
ax.set_rgrids(radii=[0, 0.025, 0.05, 0.075, 0.1, ], angle=100)
bars = ax.bar(bins[:bins_number], relfreq, width=width, bottom=0.0, color='mediumslateblue', edgecolor='black', label="Relative frequency of each step angle")

for bar in bars:
    bar.set_alpha(0.5)
plt.legend(bbox_to_anchor = [0.4, 0.01])
plt.savefig(figdir+'PolHist_StepAng_FreqDis.png', dpi=900) #This saves the first page of figures
plt.show()


"""This code makes a cartesian graph of the flight trajectory in latitude/longitude coordinates."""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
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

"""Reading in data"""
# %% load some data

dataset = 'data_sample_d4.nc'
f = Dataset(dataset, mode='r')
data = f.groups['Radar_data']   

R = data.variables['Range'][:]
Theta = data.variables['Azimuth'][:] * (np.pi/180)  # deg->rad; assume counterclockwise from East
Elevation = data.variables['Elevation'][:][0]
# ^ seems like this is height above sea level? (says unit is meters) not radar elevation angle?
h = Elevation

# radar position (lon, lat)
# seems like they are backwards in the dataset?
lat = data.variables["Longitude"][:][0]
lon = data.variables["Latitude"][:][0]
pos0_deg = (lon, lat)

f.close()


# %% compute cartesian relative positions

# first create DataFrame
df = pd.DataFrame({"r": R, "theta": Theta})

# compute relative positions
# assuming the radar elevation angle is 0
# if we had the radar scan elevation angle we could use that
df["dx"] = df["r"] * np.cos(df["theta"])
df["dy"] = df["r"] * np.sin(df["theta"])


# compute track lat/lon positions in degrees manually

# method 1 -- transform to Cartesian

# compute pos0 in x, y (m) on sphere
r_e = 6.371e6  # mean Earth radius (m)
r = r_e + h
lon0, lat0 = pos0_deg
theta = np.deg2rad(lon0)
phi = np.deg2rad(90-lat0)
x0 = r * np.sin(phi) * np.cos(theta)
y0 = r * np.sin(phi) * np.sin(theta)
z0 = r * np.cos(phi)

# compute relative distances by adding to pos0
dx = df["dx"].values  # dx in local coords (x zonal/East, y meriodonal/North)
dy = df["dy"].values  # dy " "
dz = np.zeros_like(dx)  # assume measurements are in the radar's plane, tangent to spherical Earth surface
# need to coordinate-rotate the local dx,dy to the global Cartesian coordinate system
# first rotate about x axis so local z aligns with North Pole z
# then rotate about the new z axis so local x points toward 0 deg. E
rot = Rotation.from_euler("xz", (phi, np.pi/2 + theta), degrees=False)
dxc, dyc, dzc = rot.apply(np.stack((dx, dy, dz), 1)).T
x = x0 + dxc
y = y0 + dyc
z = z0 + dzc

# back to spherical
rs = np.sqrt(x**2 + y**2 + z**2)
df["lon_manual"] = np.rad2deg(np.arctan(y/x))
# df["lat_manual"] = 90 - np.rad2deg(np.arctan(np.sqrt(x**2 + y**2)/z))
df["lat_manual"] = 90 - np.rad2deg(np.arccos(z/rs))  # ^ equiv


# method 2 -- x,y coords on sphere

# d1deg = r*2*np.pi/360  # m for one degree at equator
r = r_e
x0 = r * theta * np.sin(phi)  # r * lon_radians * cos(lat_radians)
y0 = r * (np.pi/2 - phi)  # r * lat_radians

x = x0 + dx
y = y0 + dy

df["lon_manual2"] = np.rad2deg(x/(r*np.sin(phi)))  # negating variation in lat in the track
df["lat_manual2"] = np.rad2deg(y/r)  


"""Plotting data"""

#plots the data and labels it
color = 'tab:blue'
plt.plot(df["lon_manual2"], df["lat_manual2"], 'o-', label='Flight trajectory', ms=8, color=color, markerfacecolor='gray') 

plt.plot(*pos0_deg, "*", c="gold", ms=12, mec="black", zorder=3,
        label="Radar Location \nLatitude: 40.72012 degrees North \nLongitude: 77.93085 degrees West \nElevation: 376 meters")

#these add the labels, title, legend, etc. 
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sample Bee Movement')
plt.legend()
plt.savefig(figdir+'LatLon.png', dpi=900) #This saves the first page of figures
plt.show()



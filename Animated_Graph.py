#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:54:39 2020

@author: elanacope
"""


import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.font_manager as mfm
# import math

from matplotlib import animation
# import matplotlib.pyplot as plt
from netCDF4 import Dataset


script_name = "bees-on-map_cartopy"  # fig save prefix


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

lon1=np.array(df["lon_manual"].values)
lat1= np.array(df["lat_manual"].values)


# %% define the origin and plot on map

# specify imagery we want to use
# some of these don't work in v0.17 (the latest in `defaults` channel)
# imagery = cimgt.OSM()
#imagery = cimgt.Stamen("watercolor")  # fun!
#imagery = cimgt.Stamen("terrain")
imagery = cimgt.GoogleTiles()
#imagery = cimgt.GoogleWTS()
# imagery = cimgt.MapQuestOSM()

# define projection based on the chosen imagery
proj = imagery.crs


def animateGraph():
        
    # create figure with map projection ax
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)
    ax.set_xlim((lon1.min(), lon1.max()))
    ax.set_ylim((lat1.min(), lat1.max()))
    ax.grid(True)
    
    # define map boundaries
    ax.set_extent([-77.9400, -77.9270, 40.7145, 40.7240]), #crs=proj)
    
    ax.add_image(imagery, 16)  # 2nd arg is zoom level for the downloaded tile
    # ^ 15 or 16 seems good for our scale generally
    
    # title
    ax.set_title(f"Starting location: {lat1[0]:.4f} °N, {lon1[0]:.4f} °E")


    # the plot that we will update
    ln, = ax.plot([], [], "bo-", ms=4, transform=ccrs.PlateCarree())
    londata = []
    latdata = []
    
    # bee
    bee_symbol =  "bee" # "🐝"
    prop = mfm.FontProperties(family='DejaVu Sans')
    bee = ax.annotate(bee_symbol, 
        xy=(lon1[0], lat1[0]), 
        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),  # hack: https://stackoverflow.com/questions/25416600/why-the-annotate-worked-unexpected-here-in-cartopy
        fontproperties=prop, fontsize=16, color="goldenrod",
        ha="center", va="center_baseline",
        )


    # animate()
    # Description: Every 200 ms, get speed, steering angle, and displacement estimate and update dynamic graph
    def animate(i):
        # update track
        londata.append(lon1[i])
        latdata.append(lat1[i])
        ln.set_data(londata, latdata)
        
        # update bee
        bee.set_position((lon1[i], lat1[i]))
        
        return ln, bee

    # plt.subplots_adjust(hspace = 1,wspace = 0.6)
    ani = animation.FuncAnimation(
        fig, animate, 
        frames=np.arange(lon1.size), 
        blit=True, 
        interval=200,
    )
 

    return ani


ani = animateGraph()
plt.show()



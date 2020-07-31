# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:42:52 2020

@author: zmoon
"""

# import geopandas as gpd
# import geoviews as gv
# import holoviews as hv
# import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import get_provider
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial.transform import Rotation

# transform to m
# from pyproj import Proj, transform
# p0 = Proj(init='epsg:4326')  # lon/lat deg
# pnew = Proj(init='epsg:3857')  # web mercator

# new pyproj 2 version
# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
from pyproj import Transformer
transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

script_name = "bees-on-map_bokeh"  # fig save prefix


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


# %% compute lat/lon of track positions

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


# %% bokeh plot

# transform to web mercator
# lon_wm, lat_wm = transform(p0, pnew, df["lon_manual"].values, df["lat_manual"].values)
lon_wm, lat_wm = transformer.transform(df["lat_manual"].values, df["lon_manual"].values)
df["lon_wm"] = lon_wm
df["lat_wm"] = lat_wm

# specify output file
output_file(f"{script_name}.html")

# define data source
source = ColumnDataSource(df.loc[:, ["lon_wm", "lat_wm"]])

# create figure
p = figure(x_axis_type="mercator", y_axis_type="mercator",
           active_scroll="wheel_zoom")

# add map
# p.add_tile(get_provider("OSM"))  # has more color
# p.add_tile(get_provider("CARTODBPOSITRON_RETINA"))  # prettiest one, but less color
p.add_tile(get_provider("STAMEN_TERRAIN_RETINA"))

# plot track
p.line(x="lon_wm", y="lat_wm", source=source, legend_label="track 1")

# plot radar location
# TODO: Jose wants it to be at map center (need to calculate map bounds accordingly)
lon_r, lat_r = transformer.transform(lat0, lon0)
p.circle(lon_r, lat_r, fill_color="green", size=10, legend_label="radar location")

# Jose wants to see concentric circles (100, 200, 500 m) on top
deg_per_m = 1 / (2*np.pi*6.371e6/360 * np.cos(np.deg2rad(lat0)))  # in zonal direction
for r_m in [100, 200, 500]:
    # determine what the radius should be in map proj coordinate
    # by calculating a lon value x meters to the east and then transforming it
    lon_new_deg = lon0 + deg_per_m * r_m
    lon_new, lat_new = transformer.transform(lat0, lon_new_deg)  # now in map proj coord
    r = lon_new - lon_r
    print(f"r: {r_m} m -> {r:.2f} m in the map projection")
    
    width = 2*r  # equal width, height: ellipse -> circle
    height = 2*r
    p.ellipse(x=lon_r, y=lat_r, width=width, height=height, 
              fill_color=None, line_color="blue", 
              legend_label="100, 200, 500 m")

# display legend
p.legend.location = "top_right"
p.legend.click_policy = "hide"  # I think this is a default

# show results (open the file in default web browser)
show(p)


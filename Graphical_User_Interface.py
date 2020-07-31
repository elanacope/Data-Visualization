

import netCDF4 as nc4
import math # 'math' is needed for 'sqrt' function
import matplotlib.pyplot as plt #for making graphs
import numpy as np 
import pandas as pd
import cmath
import tkinter
from tkinter import * #for the menu
from scipy import spatial
from netCDF4 import Dataset #to help with reading in netcdf
from matplotlib import rcParams
from scipy.spatial.transform import Rotation
from PIL import Image, ImageTk
# import geopandas as gpd
# import geoviews as gv
# import holoviews as hv
# import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import get_provider
#import pyproj
from windrose import WindroseAxes
# transform to m
# from pyproj import Proj, transform
# p0 = Proj(init='epsg:4326')  # lon/lat deg
# pnew = Proj(init='epsg:3857')  # web mercator

# new pyproj 2 version
# https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
from pyproj import Transformer



################################
#Part 1: This starts the graphical user interface, and creates the entry bars.
################################
root=Tk()
#Title of window
root.title('Data Aquisition')
root.geometry("800x800")
#This makes the main menu.
my_menu = Menu(root)
root.config(menu=my_menu, background="light blue" ) #This tells tkinter to use my_menu as the menu

#generating entry bars for the file generator (or radar recording eventually).
time_entry_label = Label(root, text="Length of each record (s)")
time_entry = Entry(root, width=10)
record_entry_label = Label(root, text="Number of records")
record_entry = Entry(root, width=10)
time_entry_label.pack()
time_entry.pack()
record_entry_label.pack()
record_entry.pack()


##############################
#Part 2: Create data using the netCDF file generator (w_nc4.py). 
############################


#this simulates the radar. It creates netcdf4 files. 
def click_makefiles():
    #creates variables of entered time and number of records respectively
    trget = time_entry.get()
    nrget = record_entry.get()
    
    #converts these from strings to integers
    tr = int(trget) #Note: This assumes that there is one data point per second. 
    nrint = int(nrget)
    
    # Makes a list from 0 to the number of records entered. For example, if nrint = 2, this makes the list nr = 0, 1, 2
    nr = []
    for i in range(0, nrint):
        nr.append(i)

    for i in nr:
        
        
        lon = [-77.93085]
        lat = [40.72012]
        elev = [376]
        tx_param = [3,0.08]
        rx_param = [61,10]
        df = pd.read_csv('ec_degen2015_fig1_digitized_2.csv', index_col=0, dtype=np.float64) #Dataframe
        dat = df[int(nr[i]*tr):int((nr[i]*tr)+(tr-1))].values 
        datlen = int(len(dat))
        point = np.empty(datlen)
        t = np.empty(datlen)
        xpos = np.empty(datlen)
        ypos = np.empty(datlen)
        r = np.empty(datlen)
        phi = np.empty(datlen)
        
        for n in range(datlen):
            t[n] = dat[n,0]
            xpos[n] = dat[n,3] # second elemt correspond to the position in the data_frame
            ypos[n] = dat[n,4] # second elemt correspond to the position in the data_frame
            r[n],phi[n] = cmath.polar(complex(xpos[n],ypos[n]))
        
        
        
        import netCDF4 as nc4
        savedir= '/Users/elanacope/Documents/penn_state_reu/sample_data/'
        f = nc4.Dataset(savedir+"record_"+str(nr[i]+1)+"_of_"+nrget+"_for_"+trget+"s.nc" ,'w', format='NETCDF4') #'w' stands for write
        
        tempgrp = f.createGroup('Radar_data')
        
        
        tempgrp.createDimension('lon', len(lon))
        tempgrp.createDimension('lat', len(lat))
        tempgrp.createDimension('Elev', len(elev))
        tempgrp.createDimension('Tx_param',len(tx_param))
        tempgrp.createDimension('Rx_param',len(rx_param))
        tempgrp.createDimension('T', len(t))
        tempgrp.createDimension('R', len(r))
        tempgrp.createDimension('Phi',len(phi))
        
        longitude = tempgrp.createVariable('Longitude', 'f4', 'lon')
        latitude = tempgrp.createVariable('Latitude', 'f4', 'lat')
        elevation = tempgrp.createVariable('Elevation','f4','Elev')
        tx_parameter = tempgrp.createVariable('Tx_parameter','f4','Tx_param')
        rx_parameter = tempgrp.createVariable('Rx_parameter','f4','Rx_param')
        time = tempgrp.createVariable('Time','f8','T')
        rang = tempgrp.createVariable('Range', 'f8','R')
        azimuth = tempgrp.createVariable('Azimuth','f8','Phi')
         
        
        longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
        latitude[:] = lat
        elevation[:] = elev
        tx_parameter[:] = tx_param
        rx_parameter[:] = rx_param
        time[:] = t
        rang[:] = r
        azimuth[:] = phi
        
        
        #Add global attributes
        f.description = "Sample data"
        #Add local attributes to variable instances
        longitude.units = 'degrees'
        longitude.descrption = '(+ sign)=west /// (- sign)=east'
        latitude.units = 'degrees'
        latitude.description = '(+ sign)=north /// (- sign)=south'
        elevation.units = 'meters'
        tx_parameter.units = '(KHz,us)'
        tx_parameter.description = '(PulseRepetitionRate, PulseWidth)'
        rx_parameter.units = '(MHz,-)'
        tx_parameter.units = '(SampleRate,Decimation)'
        time.units = 'seconds'
        rang.units = 'meters'
        azimuth.units = 'radians'
        
    files_made = Label(root, text = "Files are now made.")
    files_made.pack()
gobutton = Button(root, text="Make netCDF4 files", command=click_makefiles)
gobutton.pack()
        


##############################
#Part 3: Append the data files to master file.
#############################


r = []
phi = []
t = []
lat = []
lon = []
elev = []
tx_param = []
rx_param = []


def append_files():
    
        
    trget = time_entry.get()
    nrget = record_entry.get()
    
    #converts these from strings to integers
    tr = int(trget) #Note: This assumes that there is one data point per second. 
    nrint = int(nrget)
    
    # Makes a list from 0 to the number of records entered. For example, if nrint = 2, this makes the list nr = 0, 1, 2
    nr = []
    for i in range(0, nrint):
        nr.append(i)
    for i in nr:
        datadir= '/Users/elanacope/Documents/penn_state_reu/sample_data/'
        f = nc4.Dataset(datadir+"record_"+str(nr[i]+1)+"_of_"+nrget+"_for_"+trget+"s.nc", mode = 'r') #'r' stands for read
        tempgrp = f.groups['Radar_data'] 
    
    
        #This opens the netCDF file and assigns the variables that come from the file 
        
        R = list(np.array(tempgrp.variables['Range']))
        Phi= list(np.array(tempgrp.variables['Azimuth']))
        Lat = list(np.array(tempgrp.variables['Latitude']))
        Lon = list(np.array(tempgrp.variables['Longitude']))
        T = list(np.array(tempgrp.variables['Time']))
        Elev = list(np.array(tempgrp.variables['Elevation']))
        Tx_param = list(np.array(tempgrp.variables['Tx_parameter']))
        Rx_param = list(np.array(tempgrp.variables['Rx_parameter']))
    
        r.extend(R)
        phi.extend(Phi)
        t.extend(T)
        lat.extend(Lat)
        lon.extend(Lon)
        elev.extend(Elev)
        tx_param.extend(Tx_param)
        rx_param.extend(Rx_param)
    
    files_appended = Label(root, text = "Files are now appended.")
    files_appended.pack()



#makes a button to append files
apndfilebutton = Button(root, text= "Append to master file", command=append_files)
apndfilebutton.pack()



###################################
#Part 4: setting up all of the graph prefereces. 
##################################

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
plt.rcParams['ytick.labelsize']  =  16
#plt.rcParams['axes.grid']         = True
plt.rcParams['xtick.major.size']  = 5
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.size']  = 5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['lines.markersize']  = 12
figdir = '/Users/elanacope/Documents/penn_state_reu/Figures/Graphs/' #where the figures will be saved
rcParams.update({'figure.autolayout': True}) #keeps the axis labels from getting cut off



#################################
#Part 5: commands for each menu option.
#################################

#These are the functions that will be called in for each menu item that is clicked.
#The functions need to be called before the rest of the commands of the program. 


def donothing(): # This is a temporary place holder until the rest of the options are coded. 
# When this is the command for a menu option, nothing will happen. 
    pass


#These are the functions for the Graph/New Graph section of the program. 
def make_hist(): #makes a histogram of step widths

    
    #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
        
    x=np.empty(n, ) #create new arrays for x and y
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] ) #calculates and appends the values of x and y

    #Calculating step widths
    dist = np.empty((n-1, ))  # pre-allocate space for the step lengths
    for i in range(0,n-1):
        # we can use numpy's sqrt fn for numpy arrays
        dist[i] = np.sqrt( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )
    
        # scipy.spatial can do this formula for us
        dist2 = spatial.distance.euclidean((x[i], y[i]), (x[i+1], y[i+1]))
    
        # check that they are the same
        assert np.isclose(dist[i], dist2)

    #Plotting those step widths in a histogram. 
    plt.hist(dist,16,range=(0,40), histtype='bar',
    align='mid', color='lightsteelblue', label='Step Distances',edgecolor='black') #loads data and adds preferences for color, etc. 
    plt.title('Step Distances') #title of plot
    plt.xlabel('Distance traveled in 3s intervals (m)') #label x axis
    plt.ylabel('Number of occurrances') #label y axis
    plt.savefig(figdir+'Histogram.png', dpi=900) #This saves the first page of figures
    plt.show() 


def make_polar(): #makes polar plot 
    #creating polar plot
    ax=plt.subplot(111, polar=True)
    ax.plot(phi, r,'o-', label='Path 4 from (d)', ms=5, color='blue', markerfacecolor='gray') 
    ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
    ax.set_rgrids(radii=[0, 100, 200, 300, 400, 500], angle=112)
    plt.savefig(figdir+'Polar.png', dpi=900) #This saves the first page of figures
    plt.show()

def make_xy(): #makes xy plot
    

    # ^ seems like this is height above sea level? (says unit is meters) not radar elevation angle?
    h = float(elev[0])
    pos0_deg = (lon[0], lat[0])

    # first create DataFrame
    df = pd.DataFrame({"r": r, "theta": phi})

    # compute relative positions
    # assuming the radar elevation angle is 0
    # if we had the radar scan elevation angle we could use that
    df["dx"] = df["r"] * np.cos(df["theta"])
    df["dy"] = df["r"] * np.sin(df["theta"])


    # compute track lat/lon positions in degrees manually

    # method 1 -- transform to Cartesian

    # compute pos0 in x, y (m) on sphere
    r_e = 6.371e6  # mean Earth radius (m)
    R = r_e + h
    lon[0], lat[0] = pos0_deg
    Theta = np.deg2rad(lon[0])
    Phi = np.deg2rad(90-lat[0])
    x0 = R * np.sin(Phi) * np.cos(Theta)
    y0 = R * np.sin(Phi) * np.sin(Theta)
    z0 = R * np.cos(Phi)

    # compute relative distances by adding to pos0
    dx = df["dx"].values  # dx in local coords (x zonal/East, y meriodonal/North)
    dy = df["dy"].values  # dy " "
    dz = np.zeros_like(dx)  # assume measurements are in the radar's plane, tangent to spherical Earth surface
    # need to coordinate-rotate the local dx,dy to the global Cartesian coordinate system
    # first rotate about x axis so local z aligns with North Pole z
    # then rotate about the new z axis so local x points toward 0 deg. E
    rot = Rotation.from_euler("xz", (Phi, np.pi/2 + Theta), degrees=False)
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
    R = r_e
    x0 = R * Theta * np.sin(Phi)  # r * lon_radians * cos(lat_radians)
    y0 = R * (np.pi/2 - Phi)  # r * lat_radians

    x = x0 + dx
    y = y0 + dy

    df["lon_manual2"] = np.rad2deg(x/(R*np.sin(Phi)))  # negating variation in lat in the track
    df["lat_manual2"] = np.rad2deg(y/R)  


    #plots the data and labels it
    color = 'tab:blue'
    plt.plot(df["lon_manual2"], df["lat_manual2"], 'o-', label='Path 4 from (d)', ms=8, color=color, markerfacecolor='gray') 

    plt.plot(*pos0_deg, "*", c="gold", ms=12, mec="black", zorder=3,
            label="Radar Location \nLatitude: 40.72012 degrees North \nLongitude: 77.93085 degrees West \nElevation: 376 meters")

    #these add the labels, title, legend, etc. 
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Bee Movement')
    plt.legend()
    plt.savefig(figdir+'LatLon.png', dpi=900) #This saves the first page of figures
    plt.show()


def make_polarhist(): #makes polar histogram of azimuthal angles
        
    bins_number = 16  # the [0, 2pi) interval will be subdivided into this
    bins = np.linspace(-np.pi, np.pi, bins_number + 1)

    n, _, _ = plt.hist(phi, bins) # Create histogram


    plt.clf()
    width = 2 * np.pi / bins_number
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_xticklabels(['E', '', 'N', '', 'W', '', 'S', ''])
    ax.set_rgrids(radii=[0, 15, 30, 45, 60 ], angle=112)
    bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, color='mediumslateblue', edgecolor='black')
    for bar in bars:
        bar.set_alpha(0.5)
    plt.savefig(figdir+'PolarHistogram.png', dpi=900) #This saves the first page of figures
    plt.show()

def make_tseries():
    #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
    x=np.empty(n, )
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] )

    #Calculating step widths
    dist = np.empty((n-1, ))  # pre-allocate space for the step lengths
    for i in range(0,n-1):
        # we can use numpy's sqrt fn for numpy arrays
        dist[i] = np.sqrt( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )
        
        # scipy.spatial can do this formula for us
        dist2 = spatial.distance.euclidean((x[i], y[i]), (x[i+1], y[i+1]))
        
        # check that they are the same
        assert np.isclose(dist[i], dist2)


    #make time same length
    j = np.empty(n-1, )
    for i in range(0, n-1):
        j[i] = 1
    time_resized = np.empty(n-1, )
    for i in range(0, n-1):
        time_resized[i] = t[i] * j[i]

    #plots the data and labels it
    plt.plot(time_resized, dist, 'o-', label='Path 4 from (d)', ms=5, color='black', markerfacecolor='gray') 

    #these add the labels, title, legend, etc. 
    plt.xlabel('Time (s)')
    plt.ylabel('Distance traveled per 3s interval (m)')
    plt.title('Sample Bee Movement')
    plt.legend()
    plt.savefig(figdir+'tser.png', dpi=900) #This saves the first page of figures
    plt.show()


def make_georef():
    # -*- coding: utf-8 -*-
    """
    Created on Wed Jun 17 15:42:52 2020

    @author: zmoon
    """
    h = float(elev[0])
    pos0_deg = (lon[0], lat[0])

    
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

    script_name = "bees-on-map_bokeh"  # fig save prefix


    # first create DataFrame
    df = pd.DataFrame({"r": r, "theta": phi})

    # compute relative positions
    # assuming the radar elevation angle is 0
    # if we had the radar scan elevation angle we could use that
    df["dx"] = df["r"] * np.cos(df["theta"])
    df["dy"] = df["r"] * np.sin(df["theta"])


    # compute track lat/lon positions in degrees manually

    # method 1 -- transform to Cartesian

    # compute pos0 in x, y (m) on sphere
    r_e = 6.371e6  # mean Earth radius (m)
    R = r_e + h
    lon[0], lat[0] = pos0_deg
    Theta = np.deg2rad(lon[0])
    Phi = np.deg2rad(90-lat[0])
    x0 = R * np.sin(Phi) * np.cos(Theta)
    y0 = R * np.sin(Phi) * np.sin(Theta)
    z0 = R * np.cos(Phi)

    # compute relative distances by adding to pos0
    dx = df["dx"].values  # dx in local coords (x zonal/East, y meriodonal/North)
    dy = df["dy"].values  # dy " "
    dz = np.zeros_like(dx)  # assume measurements are in the radar's plane, tangent to spherical Earth surface
    # need to coordinate-rotate the local dx,dy to the global Cartesian coordinate system
    # first rotate about x axis so local z aligns with North Pole z
    # then rotate about the new z axis so local x points toward 0 deg. E
    rot = Rotation.from_euler("xz", (Phi, np.pi/2 + Theta), degrees=False)
    dxc, dyc, dzc = rot.apply(np.stack((dx, dy, dz), 1)).T
    x = x0 + dxc
    y = y0 + dyc
    z = z0 + dzc

    # back to spherical
    rs = np.sqrt(x**2 + y**2 + z**2)
    df["lon_manual"] = np.rad2deg(np.arctan(y/x))
    # df["lat_manual"] = 90 - np.rad2deg(np.arctan(np.sqrt(x**2 + y**2)/z))
    df["lat_manual"] = 90 - np.rad2deg(np.arccos(z/rs))  # ^ equiv


  
#%bokeh plot

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
    lon_r, lat_r = transformer.transform(lat[0], lon[0])
    p.circle(lon_r, lat_r, fill_color="green", size=10, legend_label="radar location")

    # Jose wants to see concentric circles (100, 200, 500 m) on top
    deg_per_m = 1 / (2*np.pi*6.371e6/360 * np.cos(np.deg2rad(lat[0])))  # in zonal direction
    for r_m in [100, 200, 500]:
        # determine what the radius should be in map proj coordinate
        # by calculating a lon value x meters to the east and then transforming it
        lon_new_deg = lon[0] + deg_per_m * r_m
        lon_new, lat_new = transformer.transform(lat[0], lon_new_deg)  # now in map proj coord
        R = lon_new - lon_r
        
        
        width = 2*R  # equal width, height: ellipse -> circle
        height = 2*R
        p.ellipse(x=lon_r, y=lat_r, width=width, height=height, 
                fill_color=None, line_color="blue", 
                legend_label="100, 200, 500 m")

    # display legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # I think this is a default

    # show results (open the file in default web browser)
    show(p)

    
def make_distseries():
    #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
    x=np.empty(n, )
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] )

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
        time_resized[i] = t[i] * j[i]


    #plots the data and labels it

    plt.plot(time_resized, total_dist, 'o-', label='Path 4 from (d)', ms=5, color='black', markerfacecolor='plum') 

    #these add the labels, title, legend, etc. 
    plt.xlabel('Time (s)')
    plt.ylabel('Total Distance Traveled (m)')
    plt.legend()
    plt.savefig(figdir+'tser_tot.png', dpi=900) #This saves the first page of figures
    plt.show()

    print(time_resized.size)


def make_SAhist(): 
    #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
    x=np.empty(n, )
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] )

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
    
def make_SA_SD_hist():
        
    #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
    x=np.empty(n, )
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] )
    
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
    
    angles_deg = np.rad2deg(angles)
    angles_north_centered = np.empty(n-1, )
    
    for i in range(0, n-1): #make this centered at north
        
        if 90 < angles_deg[i] < 180:
            angles_north_centered[i] = angles_deg[i] - 90
        else: 
            angles_north_centered[i] = angles_deg[i] + 270
            
    angles_clockwise = np.empty(n-1, )     
    for i in range(0, n-1): #make it go clockwise
        angles_clockwise[i] = 360 - angles_north_centered[i]
    
    ax = WindroseAxes.from_ax()
    ax.bar(angles_clockwise, dist, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend(decimal_places=1, title="Step Distance (m)", title_fontsize=14 )
    plt.savefig(figdir+'Step_angles_w_distance.png', dpi=900) #This saves the first page of figures
    plt.show()
    
def make_animated_graph():
        #This converts from polar coordinates to cartesian coordinates.
    n = len(r)
    x=np.empty(n, )
    y=np.empty(n, )
    for i in range(n):
        x[i]= ( math.cos(phi[i]) * r[i] )
        y[i]= ( math.sin(phi[i]) * r[i] )
    
    def animateGraph():
        # based on: https://matplotlib.org/3.2.2/api/animation_api.html
        from matplotlib import animation
        import matplotlib.font_manager as mfm
        # initialize plot
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_title("2D position estimate")
        ax1.set_ylabel(" Y displacement (m)")
        ax1.set_xlabel(" X displacement (m)")
        ax1.set_xlim((x.min(), x.max()))
        ax1.set_ylim((y.min(), y.max()))
        ax1.grid(True)
    
        # the plot that we will update
        ln, = ax1.plot([], [], "bo-")
        xdata = []
        ydata = []
    
       # bee
        bee_symbol =  "🐝" #"bee"
        prop = mfm.FontProperties(family='DejaVu Sans')
        # bee = ax1.text(0, 0, bee_symbol, fontproperties=prop, fontsize=40, ha="center", va="center")
        bee = ax1.annotate(bee_symbol, xy=(x[0], y[0]), 
                           fontproperties=prop, fontsize=20, color="goldenrod",
                           ha="center", va="center_baseline")
    
        # animate()
        # Description: Every 200 ms, get speed, steering angle, and displacement estimate and update dynamic graph
        def animate(i):
            # update track
            xdata.append(x[i])
            ydata.append(y[i])
            ln.set_data(xdata, ydata)
    
    
            # update bee
            bee.set_position((x[i], y[i]))
    
            return ln, bee
    
        # plt.subplots_adjust(hspace = 1,wspace = 0.6)
        ani = animation.FuncAnimation(
            fig, animate, 
            frames=np.arange(x.size), 
            blit=True, 
            interval=200,
        )
    
        return ani
    
    
    ani = animateGraph()
    
    plt.show()
        

    s

##################################################
#Part 7: Creating the menu bar options. 
##############################################

#create File menu
file_menu = Menu(my_menu)
my_menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="New...", command=donothing)
file_menu.add_command(label="Open", command=donothing)
file_menu.add_command(label="Save Experiment", command=donothing)
file_menu.add_command(label="Save Graphics", command=donothing)
file_menu.add_command(label="Export", command=donothing)


#create Edit menu
edit_menu = Menu(my_menu)
my_menu.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="Undo", command=donothing)
edit_menu.add_command(label="Redo", command=donothing)
edit_menu.add_command(label="Duplicate Graph", command=donothing)

#create Data menu
data_menu = Menu(my_menu)
my_menu.add_cascade(label="Data", menu=data_menu)
data_menu.add_command(label="Import Data", command=donothing)
data_menu.add_command(label="Save Data", command=donothing)
data_menu.add_command(label="Duplicate Data", command=donothing)
data_menu.add_command(label="Change Scaling", command=donothing)
data_menu.add_command(label="Redimension", command=donothing)
data_menu.add_command(label="Insert Points", command=donothing)
data_menu.add_command(label="Delete Points", command=donothing)
data_menu.add_command(label="Delete Data", command=donothing)


#create Graph menu
graph_menu = Menu(my_menu)
my_menu.add_cascade(label="Graph", menu=graph_menu)
#New Graph menu
new_graph_menu = Menu(graph_menu)
graph_menu.add_cascade(label="New Graph", menu=new_graph_menu)
new_graph_menu.add_cascade(label="Polar", command=make_polar)
new_graph_menu.add_cascade(label="Cartesian", command=make_xy)
new_graph_menu.add_cascade(label="Georeferencing", command=make_georef)
new_graph_menu.add_cascade(label="Step Length Histogram", command=make_hist)
new_graph_menu.add_cascade(label="Step Length Time Series", command=make_tseries)
new_graph_menu.add_cascade(label="Total Distance Time Series", command=make_distseries)
new_graph_menu.add_cascade(label="Orientation Histogram", command=make_polarhist)
new_graph_menu.add_cascade(label="Step Angle Histogram", command=make_SAhist)
new_graph_menu.add_cascade(label="Step Angle/Step Distance Histogram", command=make_SA_SD_hist)
new_graph_menu.add_cascade(label="Animated Graph", command=make_animated_graph)

#other options
graph_menu.add_cascade(label="Modify Graph", command=donothing)
graph_menu.add_cascade(label="Append to Graph", command=donothing)
graph_menu.add_cascade(label="Remove from Graph", command=donothing)
graph_menu.add_cascade(label="Reorder Traces", command=donothing)
graph_menu.add_cascade(label="Saved Graphs", command=donothing)


#create Calculate menu
calculate_menu = Menu(my_menu)
my_menu.add_cascade(label="Calculate", menu=calculate_menu)
calculate_menu.add_cascade(label="Path Length", command=donothing)
calculate_menu.add_cascade(label="Integrate Path", command=donothing)
calculate_menu.add_cascade(label="Differentiate Path", command=donothing)
calculate_menu.add_cascade(label="Step Lengths", command=donothing)
calculate_menu.add_cascade(label="Average Step Length", command=donothing)
calculate_menu.add_cascade(label="Time Elapsed", command=donothing)
calculate_menu.add_cascade(label="ANOVA", command=donothing)
calculate_menu.add_cascade(label="T-Test", command=donothing)
calculate_menu.add_cascade(label="Rayleigh Test", command=donothing)
calculate_menu.add_cascade(label="Turkey HSD Test", command=donothing)
calculate_menu.add_cascade(label="Kruskal-Wallis Test", command=donothing)


root.mainloop() #this is needed. Not exactly sure why. Don't delete it. 

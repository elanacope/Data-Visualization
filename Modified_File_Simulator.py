# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import cmath
from tkinter import * #for the menu
import netCDF4 as nc4

#This starts the graphical user interface.
root=Tk()
#Title of window
root.title('netCDF Generator')

#generating entry bars for file generator
time_entry_label = Label(root, text="Length of each record (s)")
time_entry = Entry(root, width=10)
record_entry_label = Label(root, text="Number of records")
record_entry = Entry(root, width=10)
time_entry_label.pack()
time_entry.pack()
record_entry_label.pack()
record_entry.pack()

    

def click_makefiles():
        
    #creates variables of entered time and number of records respectively
    trget = time_entry.get()
    nrget = record_entry.get()
    
    #converts these from strings to integers
    tr = int(trget)/1 #Note: This assumes that there is one data point per second. 
    nrint = int(nrget)
    
    # Makes a list from 0 to the number of records entered. For example, if nrint = 2, this makes the list nr = 0, 1, 2
    nr = []
    for i in range(0, nrint):
        nr.append(i)
    
    for i in nr:
        
        lon = [40.72012]
        lat = [-77.93085]
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
        
        phid = phi*180/np.pi
        
        
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
        azimuth[:] = phid
        
        
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
        azimuth.units = 'degrees'
    files_made = Label(root, text = "Files are now made.")
    files_made.pack()

makefilebutton = Button(root, text="Make netCDF4 files", command=click_makefiles)
makefilebutton.pack() #makes a button that executes the click_makefiles function when clicked


r = []
phi = []
t = []
lat = []
lon = []
elev = []
tx_param = []
rx_param = []


def ap_fi():
    
        
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
apndfilebutton = Button(root, text= "Append to master file", command=ap_fi)
apndfilebutton.pack()


root.mainloop()



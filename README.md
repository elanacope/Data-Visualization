![fancy pic could go here](fancy-pic.png)

Data visualization tools for motions of bees.

Written by Elana Cope for PSU EMS REU 2020...

## Installation

This runs in a Conda environment with Python 3.7.7.

### Dependencies

* cartopy
* netCDF4
* matplotlib
* numpy
* pandas
* cmath
* tkinter
* scipy
* PIL
* bokeh
* windrose
* pyproj


## Usage

The code for all of the graphs relies on a netCDF4 file (included), and the name of the file can be changed in the code for each figure. 

The code for the graphical user interface relies on a CSV file, and the name of that file can be changed in line 87. The folder where the netCDF4 files that the program generates should be updated in line 106, and the folder where the program reads the netCDF4 files should also be updated (line 195). Finally, the place where figures should be saved should also be updated in line 251. 

Once the code for the graphical user interface has been run, a window will open where a user is asked to input the number of recordings and the time of each recording. Integers should be entered into these bars. The time essentially specifies how many points each file will contain, and the number of recordings specifies how many files will be made. Once the files have been made, the user should click "Append to Master file." When both the "Make netCDF$ files" and the "Append to master file" buttons are clicked, there should be a verification from the program that the "files are now made" and the "files are now appended." The Graph/New Graph menu allows the user to select which kind of graph they prefer to make. None of the other menu options besides Graph/New Graph do anything; they all have a "pass" function. This code can be further modified by adding functions and modifying the "command=donothing" to "command= A new function" in lines 800-868. 


...

## Credits

Thanks to Zachary Moon ([@zmoon92](https://github.com/zmoon92)), Diego Peñaloza Aponte, José Fuentes, Julio Urbina, and Margarita López-Uribe for a lot of assistance along the way.

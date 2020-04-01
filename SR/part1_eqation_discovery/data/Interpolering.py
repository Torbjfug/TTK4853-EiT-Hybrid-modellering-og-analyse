import numpy as np

#Interpoleringsalgoritme

#Lister med dataen:


def interpolation(grid, geopotential, wind_x, wind_y, wind_z, pressure):
    #grid er antall nye z-koordinater
    new_wind_x = [0 for i in range(grid)]
    new_wind_y = [0 for i in range(grid)]
    new_wind_z = [0 for i in range(grid)]
    new_pressure = [0 for i in range(grid)]
    z_coord = np.zeros(grid+1)

    #Finner maks og min geopotensiale
    Max_gp = 2200
    Min_gp = -3.37
    #for i in range(len(geopotential)):
    #    if geopotential[i] > Max_gp:
    #        Max_gp = geopotential[i]
    #    if geopotential[i]<Min_gp:
    #        Min_gp = geopotential[i]

    Delta_gp = Max_gp-Min_gp
    #Ny spredning i z-koordinat:
    Deltaz = Delta_gp/grid
    #Nye z-koordinater:
    for i in range(grid):
        z_coord[i] = Min_gp + Deltaz*i
    z_coord[grid] = Max_gp
    geopotential = np.flip(geopotential)


    for i in range(grid):
        gp_below = np.searchsorted(geopotential, z_coord[i], side='left') - 1
        gp_above = np.searchsorted(geopotential, z_coord[i], side='left')

        if(gp_below <= 0):
            new_wind_x[i] = 0
            new_wind_y[i] = 0
            new_wind_z[i] = 0
            new_pressure[i] = 0
        else:
            weight = (z_coord[i]-geopotential[gp_below])/(geopotential[gp_above]-geopotential[gp_below])
            new_wind_x[i] = wind_x[gp_below] + weight*(wind_x[gp_above] - wind_x[gp_below])
            new_wind_y[i] = wind_y[gp_below] + weight*(wind_y[gp_above] - wind_y[gp_below])
            new_wind_z[i] = wind_z[gp_below] + weight*(wind_z[gp_above] - wind_z[gp_below])
            new_pressure[i] = pressure[gp_below] + weight*(pressure[gp_above] - pressure[gp_below])
    return new_wind_x, new_wind_y, new_wind_z, new_pressure  
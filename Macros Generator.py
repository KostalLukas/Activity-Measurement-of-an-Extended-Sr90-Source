# -*- coding: utf-8 -*-
"""
Radioactivity Macros Generator v1.0

Lukas Kostal, 8.3.2023, ICL
"""


import numpy as np
import os


# number of the simulation
n_sim = 5

# path of the build of the Geant4 simulation
path_build = '/Users/lukaskostal/Desktop/Labs/Radioactivity/Geant4/Lab_Simulation/build'

# specify first set of distances and number of events to be simulated
dist_1 = np.arange(0.5, 10, 0.5)
events_1 = np.ones(len(dist_1)) * 1

dist_2 = np.arange(10, 20, 1)
events_2 = np.ones(len(dist_2)) * 2

# specify second set of distances and number of events to be simulated
dist_3 = np.arange(20, 40, 2, )
events_3 = np.linspace(3, 12, len(dist_3)) * 1

# join the two arrays to get total distances and events for simulation
dist = np.concatenate((dist_1, dist_2, dist_3))
events = np.concatenate((events_1, events_2, events_3)) * 1e6
    
# create directory to hold the simulated data and parameters csv file
os.system(f'mkdir {path_build}/Macros_{n_sim}')
os.system(f'mkdir {path_build}/Dataset_{n_sim}')

# join the distance and events arrays to write to a .csv file
data_out = np.stack((dist, events), axis=-1)

# write the parameters for the simulation into a .csv file
hdr = 'distance (m), events'
np.savetxt(f'{path_build}/Dataset_{n_sim}/parameters.csv', data_out, header=hdr ,delimiter=',', newline='\n')

# loop over each desired distance and no of events
for i in range(0, len(dist)):
    
    # make a copy of the master macro file and save it in Macros folder
    os.system(f'cp -R {path_build}/macro_master.mac {path_build}/Macros_{n_sim}/macro_{i}.mac')
    
    # open the copied macro and append commands
    # initialise source at the descired location and simulate descired number of events
    macro = open(f'{path_build}/Macros_{n_sim}/macro_{i}.mac', 'a')
    macro.write('\n')
    macro.write(f'/lab/Source 0 0 -{dist[i]} 0 \n')
    macro.write(f'/run/beamOn {events[i]:.0f}')
    macro.close()
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:47:10 2020

@author: mamiferock
"""

"""
    This scripts extracts the information from the original data,
    and export it to pickle format. This allows to have a very light folder 
    that is easy to share. 
"""

from functions import *
from my_functions import *
from wrappers import *
import pandas as pd
import pickle
import os
from shutil import copyfile


# rootDir = '/Users/vite/navigation_system/Data'
rootDir =  '/Volumes/Seagate/Kraken/K2'
# Select animal ID and session
ID = 'A7621'
session = 'A7621-210629'
#One drive path
OneD = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data' + '/' + ID + '/' + session 

# Load the spikes and shank data
data_directory =  rootDir + '/' + ID + '/' + session 

# data_directory = '/Volumes/LaCiel/Timaios/Kilosorted/A4403/A4403-200626/A4403-200626'
spikes, shank = loadSpikeData(data_directory)

# Find the number of episodes and events
episodes = []
for file in os.listdir(data_directory):
    if 'analogin' in file:
        episodes.append('sleep')
        
events = []
for i in os.listdir(data_directory):
    if os.path.splitext(i)[1]=='.csv':
        if i.split('_')[1][0] != 'T': 
            events.append( i.split('_')[1][0])
            episodes[int(i.split('_')[1][0])]='wake'      
events.sort()

# Load the position of the animal derived from the camera tracking
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
optoloc = 2
opto_ep = loadOptoEp(data_directory, epoch=optoloc, n_channels=2, channel=1)

stim_ep = opto_ep.merge_close_intervals(100000000)


# position.index[0], stim_ep.loc[0].start, stim_ep.loc[1].start, stim_ep.loc[2].start,  
# stim_ep.loc[0].start - 400, stim_ep.loc[1].end, stim_ep.loc[2], position[-1]
# start = []
# end = []
# for i in range(stim_ep.index):
#     start.append(stim_ep.loc[i].start)
#     end.append(stim_ep.loc[i].end)
#     start.append (stim_ep.loc[i+1].end + 400)
#     end.append (stim_ep.loc[i+2].start - 400)


#create a new directory for saving the data
os.mkdir(data_directory + '/my_data')
os.mkdir(OneD)
OneD = OneD + '/my_data'
os.mkdir(One_D)
#copy XML
copyfile(data_directory + '/' + session + '.xml', OneD + '/' + session + '.xml')
# Get the time interval of the wake epoch
wake_ep = loadEpoch(data_directory, 'wake', episodes)
if 'sleep' in episodes:
    sleep_ep = loadEpoch(data_directory, 'sleep')
    with open(data_directory + '/my_data/' + 'sleep_ep' + '.pickle', 'wb') as handle:
        pickle.dump(sleep_ep, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OneD + '/' + 'sleep_ep' + '.pickle', 'wb') as handle:
        pickle.dump(sleep_ep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#save data in pickle format
        
for string, objct in zip(['opto_ep_'+str(optoloc), 'stim_ep_'+str(optoloc)], [opto_ep, stim_ep]):
    with open(data_directory + '/my_data/' + string + '.pickle', 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #save it in OneDrive
    with open(OneD + '/' + string + '.pickle', 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
for string, objct in zip(['spikes', 'shank', 'episodes', 'position', \
              'wake_ep'],
              [spikes, shank, episodes, position, wake_ep]):
    with open(data_directory + '/my_data/' + string + '.pickle', 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #save it in OneDrive
    with open(OneD + '/' + string + '.pickle', 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
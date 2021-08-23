#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:28:47 2020

@author: mamiferock
"""

from functions import *
from wrappers import *
import matplotlib.pyplot as plt
from my_functions import *
import neuroseries as nts
import pandas as pd
import pickle
import os



# def analysis(data_directory_load, dir2save_plots, ID, session):
# data_directory_load = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data/A7621/A7621-210617/'
data_directory_load = OneD+'/my_data'
dir2save_plots = data_directory_load + '/plots'
if not os.path.exists(dir2save_plots):
    os.mkdir(dir2save_plots)
session = 'A7621-210629'
# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
opto_ep = pickle.load(open(data_directory_load  + '/opto_ep_2.pickle', 'rb'))
stim_ep = pickle.load(open(data_directory_load  + '/stim_ep_2.pickle', 'rb'))
sleep_ep = pickle.load(open(data_directory_load  + '/sleep_ep.pickle', 'rb'))

"""
Sanity check
"""
plt.figure()
start_w = wake_ep.as_units('s').start.values[0]
plt.hlines(4, start_w, wake_ep.as_units('s').end.values)
for s in range(len(stim_ep)):
    begin = stim_ep.as_units('s').start[s] 
    end = stim_ep.as_units('s').end[s] 
    plt.hlines(s+1, begin, end, 'r')
plt.show()

flag = False
if flag:
    start_wake = wake_ep.start.values[0]
    opto_ep = opto_ep+start_wake
    start= stim_ep.start.values +start_wake
    end = stim_ep.end.values +start_wake
    stim_ep = nts.IntervalSet(start = start, 
                              end = end)
    plt.figure()
    plt.hlines(4, start_w, wake_ep.as_units('s').end.values)
    for s in range(len(stim_ep)):
        begin = stim_ep.as_units('s').start[s] 
        end = stim_ep.as_units('s').end[s]
        plt.hlines(s+1, begin, end, 'r')
    plt.show()

"""
Section A. Comparison of different intensities based on mean firing rate
"""

# # Get the time interval before the stimulation
# mins = 5
# baseline = nts.IntervalSet(start = opto_ep.start[0]-mins*60*1000*1000 - 1000000, 
#                          end = opto_ep.start[0]-1000000)
# # Compute the mean firing rate of the neurons for the baseline interval
# df_firnrate = computeMeanFiringRate(spikes, [baseline, 
#                                   stim_ep.loc[[0]], stim_ep.loc[[1]]], 
#                                   ["baseline", "low","high"])
# #necesitas modificarlo para cualquier numero de stims
# # Type a string to distinguish the plots of this section when saving
# label = "_meanFR_"

# # Plot the firing rate for the baseline and the different intensities
# plt.figure()
# for i in range(len(df_firnrate.index)):
#     plt.plot([df_firnrate["baseline"].values[i], df_firnrate["low"].values[i], 
#               df_firnrate["high"].values[i]], 
#              'o-', c='black', linewidth = 0.1, alpha=0.7)
# plt.title(session)
# plt.xticks([0,1,2], ["Baseline","Low","High"])
# plt.xlabel("Intensities")
# plt.ylabel("firing rate")
# plt.savefig('{}{}{}'.format(dir2save_plots, '/' + session + label + "firing", '.pdf'))

# # Compute the % of change respect to the baseline for the different intensities
# df_firinchange = (df_firnrate.loc[:,"low":"high"]*100).div(df_firnrate["baseline"], axis = 0) - 100
# df_firinchange.sort_values(by = "high", inplace = True, ascending = False)
# index = df_firinchange.index.values
# df_firinchange.index = [*spikes.keys()]
# colors = ["lightblue",  "royalblue"]
# # Use the pandas wrapper around plt.plot() to make a plot directely from the data frame
# df_firinchange.plot(marker ='o', color = colors, title = session)
# plt.xticks([*spikes.keys()], index)
# plt.ylabel("% Relative Change")
# plt.xlabel("Neurons")
# plt.savefig('{}{}{}'.format(dir2save_plots, '/' + session + label + "relativechange", '.pdf'))

# #*****************************************************************************
# # Select a threshold. All the neurons below this threshold will be selected
# threshold = -10
# condition = df_firinchange["high"] < threshold # Restric it for one intensity
# # Create a new dictionary with these neurons
# ids = df_firinchange[condition].index.values
# neurons_sel = {key:val for key, val in spikes.items() if key in ids}
# #******************************************************************************

"""
Section B. Raster plots 
"""

rasters_dic = raster.gendataU(spikes[0], sleep_ep.loc[[1]], opto_ep)

import matplotlib.patches as patches

stimduration = 100
dir2save = dir2save_plots
def rasters_opto(raster_list,  stimduration, session, dir2save, units = 'ms', \
                  linesize = 2, \
                     colorstim = "lightcyan"):
    if units == 'ms':
        scale = 1000
    elif units == 's':
        scale = 1000000
    else:
        print("wrong units input")    
    cols = 3
    rows = round(len(rasters_dic.keys())/cols)
    height = len(opto_ep)
    i=0
    fig, axes = plt.subplots(rows,cols, sharey = True, sharex = True)
    for r in range(rows):
        for c in range(cols):
            data = [item/scale for item in rasters_dic[i]] 
            axes[r][c].eventplot(data, linelengths = linesize, color='black')
            rect = patches.Rectangle((stimduration, 0), stimduration, height, facecolor = colorstim, alpha = 0.5)
            axes[r][c].add_patch(rect)
            i+=1
    fig.suptitle(session)
    fig.text(0.5, 0.04, 'time', ha='center')
    fig.text(0.04, 0.5, 'trials', va='center', rotation='vertical')
    plt.savefig(os.path.join(dir2save, "rasters_opto"))

"""
Section C. Histograms
"""
units = 'ms'
binsize = 1   
nbins = 30
colorctrl='tan'

def hist_opto():
    if units == 'ms':
        scale = 1000
    elif units == 's':
        scale = 1000000
    else:
        print("wrong units input")
    i = 0
    cols = 3
    rows = round(len(rasters_dic.keys())/cols)+1
    nbins = 40
    fig, axes = plt.subplots(rows,cols)
    for r in range(rows):
        for c in range(cols):
            print(i)
            if i>=len(rasters_dic.keys()):
                break
            else:
                data = rasters_dic[i]
                data = np.concatenate(data).ravel()
                data = [item/scale for item in data]
                axes[r][c].hist(data, bins = nbins, color = colorctrl)
                i+=1
    plt.savefig(os.path.join(dir2save, "hists_opto"))

i=12
fig, ax = plt.subplots()
data = rasters_dic[i]
data = np.concatenate(data).ravel()
data = [item/scale for item in data]
# nbins = int(array_hist[-1]*2/binsize)
ax.hist(data, bins = 40, color = colorctrl)

        
        
 
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
from matplotlib.gridspec import GridSpecFromSubplotSpec


# def analysis(data_directory_load, dir2save_plots, ID, session):
# data_directory_load = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data/A7621/A7621-210617/'
data_directory_load = OneD+'/my_data'
dir2save_plots = data_directory_load + '/plots'
if not os.path.exists(dir2save_plots):
    os.mkdir(dir2save_plots)
session = 'A7621-210622'
# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
opto_ep = pickle.load(open(data_directory_load  + '/opto_ep.pickle', 'rb'))
stim_ep = pickle.load(open(data_directory_load  + '/stim_ep.pickle', 'rb'))

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

# Get the time interval before the stimulation
mins = 5
baseline = nts.IntervalSet(start = opto_ep.start[0]-mins*60*1000*1000 - 1000000, 
                         end = opto_ep.start[0]-1000000)
# Compute the mean firing rate of the neurons for the baseline interval
df_firnrate = computeMeanFiringRate(spikes, [baseline, 
                                  stim_ep.loc[[0]], stim_ep.loc[[1]]], 
                                  ["baseline", "low","high"])
#necesitas modificarlo para cualquier numero de stims
# Type a string to distinguish the plots of this section when saving
label = "_meanFR_"

# Plot the firing rate for the baseline and the different intensities
plt.figure()
for i in range(len(df_firnrate.index)):
    plt.plot([df_firnrate["baseline"].values[i], df_firnrate["low"].values[i], 
              df_firnrate["high"].values[i]], 
             'o-', c='black', linewidth = 0.1, alpha=0.7)
plt.title(session)
plt.xticks([0,1,2], ["Baseline","Low","High"])
plt.xlabel("Intensities")
plt.ylabel("firing rate")
plt.savefig('{}{}{}'.format(dir2save_plots, '/' + session + label + "firing", '.pdf'))

# Compute the % of change respect to the baseline for the different intensities
df_firinchange = (df_firnrate.loc[:,"low":"high"]*100).div(df_firnrate["baseline"], axis = 0) - 100
df_firinchange.sort_values(by = "high", inplace = True, ascending = False)
index = df_firinchange.index.values
df_firinchange.index = [*spikes.keys()]
colors = ["lightblue",  "royalblue"]
# Use the pandas wrapper around plt.plot() to make a plot directely from the data frame
df_firinchange.plot(marker ='o', color = colors, title = session)
plt.xticks([*spikes.keys()], index)
plt.ylabel("% Relative Change")
plt.xlabel("Neurons")
plt.savefig('{}{}{}'.format(dir2save_plots, '/' + session + label + "relativechange", '.pdf'))

#*****************************************************************************
# Select a threshold. All the neurons below this threshold will be selected
threshold = -10
condition = df_firinchange["high"] < threshold # Restric it for one intensity
# Create a new dictionary with these neurons
ids = df_firinchange[condition].index.values
neurons_sel = {key:val for key, val in spikes.items() if key in ids}
#******************************************************************************

"""
Section B. Raster plots 
"""

# Select the time interval of the desired stimulation epoch  
intensity = 'High'
interval = stim_ep.loc[[2]]
pre = 1*6e7 #time before the stimulation in us
post = 2*6e7 #time since the stimulation in us
# Use this method from the class raster to get the data for the raster plot 
raster_list = raster.gendata(neurons_sel, pre, post, [stim_ep['start'][2]])
# Get the total duration of the stimulation
stimduration = interval.tot_length('s')
ephysplots.raster(raster_list,  stimduration, neurons_sel.keys(), session, dir2save_plots, intensity_label=intensity, binsize = 10)

#
# Select the time interval of the desired stimulation epoch  

# Get the total duration of the stimulation
stimduration = interval.tot_length('s')
frates, rasters, bins, stim_duration = computeRasterOpto(neurons_sel, stim_ep, 1000)
ephysplots.raster(raster_list,  1, neurons_sel.keys(), session, dir2save_plots, intensity_label=intensity, binsize = 10)

plt.figure()
plt.eventplot(rasters[1].index.values)

groups = np.array_split(list(neurons_sel.keys()), 2)
for i, neurons in enumerate(groups):        
    figure()
    count = 1
    for k,n in enumerate(neurons):
        ax = subplot(int(np.sqrt(len(neurons)))+1,int(np.sqrt(len(neurons)))+1,count)
        subgs = GridSpecFromSubplotSpec(2, 2, ax)
        subplot(subgs[:,0], projection = 'polar')
        plot(tuning_curves[n])        
        subplot(subgs[0,1])        
        bar(frates[n].index.values, frates[n].values, np.diff(frates[n].index.values)[0])
        #axvline(20000000)
        #axvline(40000000)
        title(n)
        subplot(subgs[1,1])
        plot(rasters[n], '.', markersize = 0.24)
        count+=1
        gca().set_xticklabels([])
        #axvline(20000000)
        #axvline(40000000)
fig = plt.figure()
for k,n in enumerate(neurons_sel):
    fig.add_subplot(3,2,k+1)
    plt.plot(rasters[n], '.', markersize = 0.24)

"""
Section C. Matrix
"""
pre_stim_m = 1*60*1000*1000 #time before the stimulation in us
post_stim_m = 2*60*1000*1000 #It must be the double of pre
step = 10*1000*1000
label_epochs = ["Low", "High"]

for lap, label in zip ([*stim_ep['start']], label_epochs):
    #determine the mean firing rate previos to the stimulation epoch per neuron
    intervalb = nts.IntervalSet(start = lap -pre_stim_m  -1000*1000, end = lap -1000*1000)
    FRbase = computeMeanFiringRate(spikes, [intervalb], ['base'])
    #generate data frame
    matrix_data = matrix.gendatab(spikes, pre_stim_m, post_stim_m, [lap], step, FRbase)
    #Order the neurons based on higher firing rate during the stimulation epoch
    dic = {key:val for key, val in enumerate(matrix_data.loc[0:120000000].sum())}
    order = pd.DataFrame(data=[dic.keys(),dic.values()]).transpose().sort_values(
            by=1, ascending=False)[0]
    matrix_data = matrix_data[order]
    matrix_data.rename(index = lambda s: int(s/1000000), inplace = True)
    # This function helps you to plot the result
    ephysplots.matrix(pre_stim_m, post_stim_m, matrix_data, session, dir2save_plots, label)

"""
Section D. Distribution plots
"""
stim_duration = 2*6e7 # Stimulus duration in us
# Restrict the data from the matrix to the stimulation epoch
df_stim = matrix_data.loc[0:stim_duration]
#  Organize the data from df_stim to make it easy to plot with seaborn
df_stim_sns = pd2seaborn(df_stim, spikes.keys(), shank)
# Create distribution plots of the firing rate as % of change for neurons and shanks 
ephysplots.distplot(df_stim_sns, spikes.keys(), label, session, dir2save_plots)

    


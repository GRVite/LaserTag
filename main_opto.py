#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:40:41 2019

@author: grvite
"""


import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
from wrappers import *
from functions import *
from scipy.stats import pearsonr
from my_functions import *



#rootDir = '/media/3TBHDD/Data'
# rootDir = '/Users/vite/navigation_system/Data'
# rootDir = '/Volumes/LaCie/Timaios/Kilosorted'
ID = 'A6100'
session = 'A6100-201106'
episodes = df[df['Session']=='A4405-200312']["Episodes"].values.tolist()
episodes = episodes[0].split(',')

# data_directory = rootDir + '/' + ID + '/' + session + '/' + session
data_directory = '/Users/vite/OneDrive - McGill University/PeyracheLab' + '/' + ID + '/' + session 
events = []
for i in os.listdir(data_directory):
    if os.path.splitext(i)[1]=='.csv':
        if i.split('_')[1][0] != 'T': 
            print(i)
            events.append( i.split('_')[1][0])

#
# spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
# position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
# wake_ep                             = loadEpoch(data_directory, 'wake', episodes)
# if "sleep" in episodes:
#     sleep_ep                             = loadEpoch(data_directory, 'sleep')                    
# ttl_track, ttl_opto_start, ttl_opto_end = loadTTLPulse2(os.path.join(data_directory, session+'_0_analogin.dat'), 2)
# ttl_track = nts.Ts(ttl_track.index.values, time_units = 's')
# ttl_opto_start = nts.Ts(ttl_opto_start.index.values, time_units = 's')
# ttl_opto_end = nts.Ts(ttl_opto_end.index.values, time_units = 's')
# opto_ep = nts.IntervalSet(start = ttl_opto_start.index.values, end = ttl_opto_end.index.values)
# stim_ep=manage.optoeps(ttl_opto_start, ttl_opto_end) #Load main stim epochs
#
"""
Comparison of different intensities based on mean firing rate
"""
plotk = "_meanFR_"
#A by mean firing rate
#baseline = nts.IntervalSet(start=wake_ep.start, end=opto_ep.start[0]-1000000)
mins = 2
baseline=nts.IntervalSet(start = stim_ep.start.iloc[0]-mins*60*1000*1000 - 1000000, 
                         end=stim_ep.start.iloc[0]-1000000)
FR=computeMeanFiringRate(spikes, [baseline, 
                                  stim_ep.iloc[[0]], stim_ep.iloc[[1]], stim_ep.iloc[[2]]], 
                                  ["baseline", "low","med","high"])
plt.figure()
for i in range(len(FR.index)):
    plt.plot([FR["baseline"].values[i], FR["low"].values[i], FR["med"].values[i], FR["high"].values[i]], 
             'o-', c='black', linewidth = 0.1, alpha=0.7)
plt.title(session)
plt.xticks([0,1,2,3], ["Baseline","Low","Medium","High"])
plt.xlabel("Intensities")
plt.ylabel("firing rate")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "firing", '.pdf'))

#as a % of change respect to the baseline
FRp=(FR.loc[:,"low":"high"]*100).div(FR["baseline"], axis=0)-100
FRp.sort_values(by="high", inplace=True, ascending = False)
index = FRp.index.values
FRp.index = [*spikes.keys()]
colors = ["lightblue", "deepskyblue", "royalblue"]
FRp.plot(marker='o', color =colors, title=session)
plt.xticks([*spikes.keys()], index)
plt.ylabel("Relative Change")
plt.xlabel("Neurons")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "relativechange", '.pdf'))
FRp.index = index

plt.figure()
for i in range(len(FR.index)):
    b=FR["baseline"].values[i]
    plt.plot([FRp["low"].values[i], FRp["med"].values[i], FRp["high"].values[i]], 
             'o-', c='black', linewidth = 0.1, alpha=0.7)
plt.title(session)
plt.xticks([0,1,2], ["Low","Medium","High"])
plt.xlabel("Intensities")
plt.ylabel("Relative change")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "relativechangeb", '.pdf'))

#select neurons for a given cutoff
cutoff = -10
condition = ((FRp["low"]<cutoff)&(FRp["med"]<cutoff)&(FRp["high"]<cutoff)).values
neurons = FRp[condition].index.values
nspikes = {key:val for key, val in spikes.items() if key in neurons}
#try it for high intensity
condition = FRp["high"]<cutoff
neurons = FRp[condition].index.values # IDs Neurons selected
nspikesh = {key:val for key, val in spikes.items() if key in neurons}

"""
#deleting neurons

#Ammend the order
ini_list = [*range(len(spikes.keys()))]
spikesH = dict(zip(ini_list, list(spikes.values()))) 
"""

"""
Rasters

import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')
"""

intensity = " High intensity"
neurons_sel = nspikesh
interval = stim_ep.loc[[2]]
pre = 2*60*1000*1000 #time before the stimulation
post = 4*60*1000*1000 #time since the stimulation
lista = raster.gendata(neurons_sel, pre, post, [stim_ep['start'][2]])
stimduration = interval.tot_length('s')
binsize = 5
#data, edges = raster.histplot(lista, "test", "s", binsize, stimduration, session)


ylabel = "Firing Rate"
linesize=0.5
cestim="lightcyan"
cctrl="tan"
units = 's'
width = stimduration
if units == 'ms':
    scale = 1000
elif units == 's':
    scale = 1000000
else:
    print("wrong units input")
lista = [i/scale for i in lista]
array = np.concatenate(lista).ravel()
nbins=int(array[-1]*2/binsize)


fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize = [16,12])
#fig = plt.figure()
#ax1 = fig.add_subplot(1,2,1)
ax1.eventplot(lista, linelengths = linesize, color='black')
left, bottom, height = (0, 0.5, len(lista))
rect = plt.Rectangle((left, bottom), width, height, facecolor=cestim, alpha=0.5)
ax1.add_patch(rect)
ax1.set_ylabel('Neurons')
ax1.set_yticks([*range(len(neurons_sel))])
ax1.set_yticklabels( {v:k for k,v in enumerate(neurons_sel.keys())} ) 
# ax1.set_y
ax1.legend(["ChR2"])
# ax1.set_title(session)
ax1.set_frame_on(False)
data, edges = np.histogram(array,nbins)
ax2.hist(array, bins=nbins, color =cctrl)
ax2.plot([0, stimduration],[data.max(), data.max()], linewidth=7, color=cestim)
ax2.set_xlabel("{}{}".format('Time (',units+')'))
ax2.set_ylabel(ylabel)
ax2.set_frame_on(False)
plt.show()
plt.suptitle(session + intensity)
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', "raster_S3" + intensity, '.pdf'))

#*****************************************************************************
#Just one neuron
neuron = 7
neuron = {neuron: spikes[neuron]}
intensity = " High intensity"
interval = stim_ep.loc[[2]]
pre = 10*60*1000*1000 #time before the stimulation
post = 4*60*1000*1000 #time since the stimulation
lista = raster.gendata(neuron, pre, post, [stim_ep['start'][2]])
stimduration = interval.tot_length('s')
binsize=5
#data, edges = raster.histplot(lista, "test", "s", binsize, stimduration, session)


ylabel = "Firing Rate"
linesize=0.5
cestim="lightcyan"
cctrl="tan"
units = 's'
width = stimduration
if units == 'ms':
    scale = 1000
elif units == 's':
    scale = 1000000
else:
    print("wrong units input")
lista = [i/scale for i in lista]
array = np.concatenate(lista).ravel()
nbins=int(array[-1]*2/binsize)


fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize = [16,12])
#fig = plt.figure()
#ax1 = fig.add_subplot(1,2,1)
ax1.eventplot(lista, linelengths = linesize, color='black')
left, bottom, height = (0, 0.5, len(lista))
rect = plt.Rectangle((left, bottom), width, height, facecolor=cestim, alpha=0.5)
ax1.add_patch(rect)
ax1.set_ylabel('Neuron ' + str(*neuron.keys()))
ax1.set_yticks([])
ax1.legend(["ChR2"])
# ax1.set_title(session)
ax1.set_frame_on(False)
data, edges = np.histogram(array,nbins)
ax2.hist(array, bins=nbins, color =cctrl)
ax2.plot([0, stimduration],[data.max(), data.max()], linewidth=7, color=cestim)
ax2.set_xlabel("{}{}".format('Time (',units+')'))
ax2.set_ylabel(ylabel)
ax2.set_frame_on(False)
plt.show()
plt.suptitle(session + intensity)
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', "raster n_" +  str(*neuron.keys()), '.pdf'))


"""
matrix
"""
#A by firing rate
pre = 2*60*1000*1000
post = 4*60*1000*1000
stepmin = 20
step=stepmin*1000*1000
labels = ["Low", "Med", "High"]
# rango = [*arange(-80,240,40)]
rango = [*arange(-int(pre/1000/1000),int(post/1000/1000),40)]

for lap, l in zip ([*stim_ep['start']], labels):
    spikes_df = matrix.gendata(spikes, pre, post, [lap], "s", step)
    print(spikes.keys())
    #Order based on higher firing rate during the stimulation epoch
    dic = {key:val for key, val in enumerate(spikes_df.loc[0:(post/2)].sum())}
    print(dic)
    order = pd.DataFrame(data=[dic.keys(),dic.values()]).transpose().sort_values(
            by=1, ascending=False)[0]
    spikes_df = spikes_df[order]
    spikes_df.rename(index = lambda s: int(s/1000000), inplace = True)
    #plot
    lista=[]
    for i in arange(-120,240,stepmin):
        if i in rango:
            lista.append(i)
        else:
            lista.append("")
    fig = plt.figure(figsize = (20, 15))
    ax=fig.add_subplot(111,label="1")
    ax = sns.heatmap(spikes_df.transpose(), cmap= "coolwarm", xticklabels=lista)
    plt.title(session + " {}".format(l))
    plt.show()
    plt.tight_layout()
    plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_matrix_"+l, '.pdf'))

#B as a % of change respect to the baseline
pre = 2*60*1000*1000
post = 4*60*1000*1000 #It must be the double of pre
step=10*1000*1000
label = ["Low", "Med", "High"]

for lap, l in zip ([*stim_ep['start']], label):
    #determine the mean firing rate previos to the stimulation epoch per neuron
    intervalb = nts.IntervalSet(start = lap -pre -1000*1000, end = lap -1000*1000)
    FRbase=computeMeanFiringRate(spikes, [intervalb], ['base'])
    #generate data frame
    spikes_dfb = matrix.gendatab(spikes, pre, post, [lap], step, FRbase)
    #Order the neurons based on higher firing rate during the stimulation epoch
    dic = {key:val for key, val in enumerate(spikes_dfb.loc[0:120000000].sum())}
    order = pd.DataFrame(data=[dic.keys(),dic.values()]).transpose().sort_values(
            by=1, ascending=False)[0]
    spikes_dfb = spikes_dfb[order]
    spikes_dfb.rename(index = lambda s: int(s/1000000), inplace = True)
    #plot
    rango = [*arange(-int(pre/1000/1000),int(post/1000/1000),40)]
    xticklabels = []
    for i in arange(-int(pre/1000/1000),int(post/1000/1000), 10):
        if i in rango:
            xticklabels.append(i)
        else:
            xticklabels.append("")
    fig = plt.figure(figsize = (20, 15))
    ax=fig.add_subplot(111,label="1")
    ax = sns.heatmap(spikes_dfb.transpose(), cmap= "coolwarm", xticklabels = xticklabels, center = 0, cbar_kws={'label': 'firing rate as % change'})
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron")
    ax2=ax.twinx()
    ax2.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    ax2.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    plt.title(session+" {}".format(l))
    plt.show()
    plt.tight_layout()
    plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_matrixb_"+l , '.pdf'))

    
    """
    S h a n k
    """
    
    """
    Clusters
    """
    
    lut = dict(zip([1,2,3,4], ['lightblue', 'steelblue', 'royalblue', 'midnightblue']))
    shanks = pd.Series(shank)
    rcolors = shanks.map(lut)
    g = sns.clustermap(spikes_dfb.transpose(), 
                        col_cluster = False, 
                        xticklabels=xticklabels, 
                        cmap= "coolwarm", 
                        center = 0, 
                        dendrogram_ratio=(.1, .2),
                        row_colors=rcolors,
                        cbar_kws={'label': 'firing rate as % change'},
                        )
    sns.despine(right = True)
    g.fig.suptitle(session + " Clustermap " + l)
    ax = g.ax_heatmap
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neurons")
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color("white")
    ax.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    ax.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    ax2=ax.twinx()
    ax2.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1)
    ax2.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1)
    
    g.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_clustermap_"+l , '.pdf'))
    
    estim_dur = int(post/2/1000/1000)
    df_estim = spikes_dfb.loc[0:estim_dur]
    sn_data = []
    for i in spikes.keys(): 
        sn_data.append(df_estim[i].values.flatten())
    sn_data = np.concatenate(sn_data).ravel()
    stim = df_estim.index
    sn_neurons = []
    for i in spikes.keys():
        for j in range(len(stim)):
            sn_neurons.append(int(i))
    sn_shanks = []
    for i in shanks:
        for j in range(len(stim)):
            sn_shanks.append(int(i))
    sn_time = []
    for i in spikes.keys():
        sn_time.append(stim)
    sn_time = np.concatenate(sn_time).ravel()
    sn_df = pd.DataFrame(np.stack((sn_data, sn_time, sn_neurons, sn_shanks), axis =1), columns = ["firing rate as % change","time", "neurons", "shanks"])
    
    from matplotlib import colors as mcolors
    hsv = []
    for i in ['lightblue', 'steelblue', 'royalblue', 'midnightblue']:
        hsv.append(mcolors.to_hex(i))
    #palette = sns.xkcd_palette(['lightblue', 'steelblue', 'royalblue', 'midnightblue'])
    palette = sns.color_palette(hsv)
    # sns.set(style="whitegrid")
    #sns.set_context(None)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    g = sns.boxenplot (x = 'neurons', y = "firing rate as % change",  hue = 'shanks', palette=palette, data = sn_df, ax= ax1)
    # g.set(xticklabels=[*spikes.keys()])
    g.set(xticks = [])
    ax2 = fig1.add_subplot(1, 2, 2)
    sns.barplot (x = 'shanks', y = "firing rate as % change",  palette = palette, data = sn_df, ax= ax2)
    plt.suptitle(session + " " + l)
    plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_clustermap_"+l , '.pdf'))


"""
Distribution plots
"""
estim_dur = int(post/2/1000/1000)
df_estim = spikes_dfb.loc[0:estim_dur]
sn_data = []
for i in spikes.keys(): 
    sn_data.append(df_estim[i].values.flatten())
sn_data = np.concatenate(sn_data).ravel()
stim = df_estim.index
sn_neurons = []
for i in spikes.keys():
    for j in range(len(stim)):
        sn_neurons.append(int(i))
sn_shanks = []
for i in shanks:
    for j in range(len(stim)):
        sn_shanks.append(int(i))
sn_time = []
for i in spikes.keys():
    sn_time.append(stim)
sn_time = np.concatenate(sn_time).ravel()
sn_df = pd.DataFrame(np.stack((sn_data, sn_time, sn_neurons, sn_shanks), axis =1), columns = ["firing rate as % change","time", "neurons", "shanks"])

from matplotlib import colors as mcolors
hsv = []
for i in ['lightblue', 'steelblue', 'royalblue', 'midnightblue']:
    hsv.append(mcolors.to_hex(i))
#palette = sns.xkcd_palette(['lightblue', 'steelblue', 'royalblue', 'midnightblue'])
palette = sns.color_palette(hsv)
# sns.set(style="whitegrid")
#sns.set_context(None)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 2, 1)
g = sns.boxenplot (x = 'neurons', y = "firing rate as % change",  hue = 'shanks', palette=palette, data = sn_df, ax= ax1)
# g.set(xticklabels=[*spikes.keys()])
ax2 = fig1.add_subplot(1, 2, 2)
sns.barplot (x = 'shanks', y = "firing rate as % change",  palette = palette, data = sn_df, ax= ax2)
plt.suptitle(session + " " + l)
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_clustermap_"+l , '.pdf'))



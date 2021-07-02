#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:13:42 2020

@author: vite
"""

import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
from functions import *
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import pickle5 as pickle
from wrappers import *
import math

"""
A. LOAD DATA
"""

# def analysis(data_directory_load, dir2save_plots, ID, session):
# data_directory_load = '/Users/vite/navigation_system/Data/A6100/A6100-201026'
data_directory_load = '/Users/vite/OneDrive - McGill University/PeyracheLab/Data/A7701/A7701-210216/my_data'
# data_directory_load = OneD
# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
# opto_ep = pickle.load(open(data_directory_load  + '/opto_ep.pickle', 'rb'))
# stim_ep = pickle.load(open(data_directory_load  + '/stim_ep.pickle', 'rb'))

dir2save_plots = data_directory_load + '/plots'
if not os.path.exists(dir2save_plots):
    os.mkdir(dir2save_plots)

"""
All, whole recording
"""
from functions import computeAngularTuningCurves
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
from functions import smoothAngularTuningCurves
tuning_curves = smoothAngularTuningCurves(tuning_curves, 30, 5)
numNeurons = len (spikes.keys())
if numNeurons < 30:
    #Determine the number of raws
    raws = round(len(spikes)/5)
    plt.figure(figsize=(40,200))
    for i, n in enumerate(spikes.keys()):
        ax=plt.subplot(5,raws+1,i+1, projection = 'polar')
        plt.plot(tuning_curves[n], color = 'darkorange')
        plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=25)
    plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
    plt.show()
    plt.savefig(data_directory_load + '/plots' + '/HD_whole_recording.pdf')
else:
    groups = int(ceil(numNeurons/30))
    start = 0
    stop = 30
    it = [*range(groups)]
    for g in it:
        selection = range(start,stop)
        print("selection",selection)
        plt.figure(figsize=(40,200))
        for i, n in enumerate(selection):
            ax=plt.subplot(5,6,i+1, projection = 'polar')
            plt.plot(tuning_curves[n], color = 'darkorange')
            plt.title('Neuron' + ' ' + str(i) + ' shank_' +str(shank[n]) + '(' + str(n) + ')', 
                      loc ='center', pad=25)
        plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
        plt.show()
        plt.savefig(data_directory_load + '/plots' + '/HD_whole_recording' + str(g) +'.pdf')
        start+= 30
        stop+=30
        if g == it[-2]:
            stop  = numNeurons
     
# 
ids, stat = findHDCells(tuning_curves, z = 20, p = 0.05, m=1)
neurons_sel = {key:val for key, val in spikes.items() if key in ids}
tuning_curves_sel = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves_sel = smoothAngularTuningCurves(tuning_curves, 10, 3)
#
raws = round(len(spikes)/5)
plt.figure(figsize=(40,200))
for i, n in enumerate(neurons_sel.keys()):
    ax=plt.subplot(5,raws+1,i+1, projection = 'polar')
    plt.plot(tuning_curves_sel[n], color = 'darkorange')
    plt.title('Neuron' + ' ' + str(n) , loc ='center', pad=25)
plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
plt.show()
plt.savefig(data_directory_load + '/plots' + '/HD_sel.pdf')

    
"""
All, selected interval
"""

interval = nts.IntervalSet(start = wake_ep.loc[0,'start'], 
                           end=wake_ep.loc[0,'start']+10*60e6)
tuning_curves = computeAngularTuningCurves(neurons_sel, position['ry'], interval, 60)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 10, deviation = 3)

if numNeurons < 30:
    #Determine the number of raws
    raws = round(len(spikes)/5)
    plt.figure(figsize=(40,200))
    for i, n in enumerate(selection):
        ax=plt.subplot(5,raws+1,i+1, projection = 'polar')
        plt.plot(tuning_curves[n], color = 'darkorange')
        plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=25)
    plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
    plt.show()
    plt.savefig(data_directory_load + '/plots' + '/HD_10min.pdf')
else:
    groups = int(ceil(numNeurons/30))
    start = 0
    stop = 30
    it = [*range(groups)]
    for g in it:
        selection = range(start,stop)
        print("selection",selection)
        plt.figure(figsize=(40,200))
        for i, n in enumerate(selection):
            ax=plt.subplot(5,6,i+1, projection = 'polar')
            plt.plot(tuning_curves[n], color = 'darkorange')
            plt.title('Neuron' + ' ' + str(i) + ' shank_' +str(shank[n]), loc ='center', pad=25)
        plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
        plt.show()
        plt.savefig(data_directory_load + '/plots' + '/HD_10min' + str(g) +'.pdf')
        start+= 30
        stop+=30
        if g == it[-2]:
            stop  = numNeurons

#store data
for string, objct in zip(['spikes_HDCells', 'tuning_curves','tuning_curves_HDcells',\
              'wake_ep'],
              [neurons_sel, tuning_curves, tuning_curves_sel]):
    #pickle it!          
    with open(os.path.join(data_directory_load, string + '.pickle'), 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
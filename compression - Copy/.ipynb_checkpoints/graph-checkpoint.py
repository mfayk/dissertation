#!/usr/bin/env python

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
#from OctCorrection import *
#from ImageProcessing import *

from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re
import pickle
from PIL import Image
import pandas as pd
#import cv2
from tifffile import imsave, imread
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rawpy
import imageio

#from OCT_reader import get_OCTSpectralRawFrame

from oct_converter.readers import FDS


from struct import unpack

import textwrap


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

    
def app_minimum_throughput_single(df, labels, xlab, ylab, ylim, title):
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(5,8))
    plt.close(1)
    #seaborn.set(font_scale = 2)
    for ax, graph_df,tol,axis in zip(axs, [df0, df10, df20],[0,10,20], [1,2,3]):
        graph_df.set_index('strategy').plot(kind="bar", stacked=True, color=['skyblue', 'pink'])
        ax.set(xlabel=xlab, ylabel=ylab)
        ax.set_title(title, weight='bold', fontsize=18)
        if(tol != 0):
            ylim=15
        ax.set_ylim(0, ylim)
        #ax.legend(title='Timer', loc='upper left', bbox_to_anchor=(1,1))
        ax.margins(y=0.3) # make room for the labels
        for bar in ax.patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
            ax.annotate(format(bar.get_height(), '.3f'),
                           (bar.get_x() + bar.get_width() / 2,
                            bar.get_height() / 2), ha='center', va='center',
                           size=10, xytext=(0, 5), rotation=30, weight='bold',
                           textcoords='offset points')
    # for bars in plot.containers:
    #     plot.bar_label(bars, fmt='%.3f', label_type='center')
    #plt.tight_layout()
        ax.set_xticklabels(labels, fontsize=14)
        wrap_labels(ax, 10)
        plt.tight_layout()
        plt.show()
        #plot.figure.savefig(outfile) (edited) 

    
    
#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']

df = pd.read_csv('city_base_lossy.csv')
#df = pd.read_csv('data/bioFilm_tiff_lossy_threads.csv')

diff0 = pd.read_csv('city_base_lossy.csv')







#app_minimum_throughput_single(df, labels, xlab, ylab, ylim, title)






#name = "/scratch/mfaykus/BioFilm/10.20.22/ATLC3_Trial1/10.20.2022.ATLC30000.tif"
#name = "/scratch/mfaykus/BioFilm/11.17.22/ATLC8b/ATLC8b.2days.11.17.220001.tif"
#name = "/scratch/mfaykus/BioFilm/3.10.23/ATLC3/ATLC3.3.10.23_0002.tif"
#name = "/scratch/mfaykus/BioFilm/3.10.23/HB/HB.3.10.230003.tif"
#name = "/scratch/mfaykus/BioFilm/3.15.23/HB2.3.15.23/HB2.3.15.230004.tif"
#name = "/scratch/mfaykus/BioFilm/3.29.23/ATLC3/Static/ATLC3_Static_3_29_23_day70005.tif"
#name = "/scratch/mfaykus/BioFilm/3.29.23/ATLC3/Transfer/ATLC3_Transfer_3_29_23_day7_0006.tif"
#name = "/scratch/mfaykus/BioFilm/3.29.23/HB2/Static/HB2_Static_3_29_23_Day7_0007.tif"
#name = "/scratch/mfaykus/BioFilm/9.30.22/ATCL8b.Day3.Trial1/ATCL8b.day3.Trial10008.tif"
#name = "/scratch/mfaykus/BioFilm/9.30.22/Control/Control.Trial10009.tif"
#name = "/scratch/mfaykus/BioFilm/9.30.22/H2.Day3.Trial1/Common Trial0010.tif"
#name = "/scratch/mfaykus/BioFilm/9.30.22/H2.Day3.Trial2/H2.day3.Trial20011.tif"


#print(df.loc[df['filename'] == name,["CR"]])
difference =  ((df.loc[:,["CR"]] - diff0.loc[:,["CR"]])/diff0.loc[:,["CR"]])*100
#difference =  ((df.loc[df['filename'] == name,["CR"]] - diff0.loc[diff0['filename'] == name,["CR"]])/diff0.loc[diff0['filename'] == name,["CR"]])*100
print("difference")
print(difference.mean())


print(df.groupby(['compressor'])['CR'].max())

print(df.groupby(['diff'])['CR'].max())

print(df.groupby(['compressor'])['ssim'].max())

print(df.groupby(['diff'])['ssim'].max())

print(df.groupby(['diff'])['CR'])

#pd.set_option("display.max_rows", 10000)
#pd.set_option("display.expand_frame_repr", True)
#pd.set_option('display.width', 10000)

#print(df.loc[(df['CR'] <= 1 ),["diff"]])
#print(df.loc[(df['CR'] <= 1 ),["filename"]])

#line_plot = sns.lineplot(data=df, x="bound", y="CR", hue = "diff")
#line_plot = sns.lineplot(data=df.query("compressor =='zfp'"), x="bound", y="cBW", hue = "diff")
#line_plot.set(xlabel='bound', ylabel='Compression Ratio')
#line_plot.legend([],[], frameon=False)
#line_plot.plot(legend=False)
#line_plot.set(xlabel='bound', ylabel='Compression Ratio')
xlab = "Error Bound"
ylab = "Compression  Ratio"
title = "SZ: Pre-processing Compression Ratio (CR)"



#bar_plot = sns.barplot(data=df, x="bound", y="CR", hue = "diff")
bar_plot = sns.barplot(data=df.query("compressor =='sz3'"), x="bound", y="psnr")
#bar_plot.set_xticklabels(bar_plot.get_xticklabels(), fontsize=4.5)

#line_plot.axhline(0, color = 'r')
#plt.ylim(0, 75)

bar_plot.set_xticklabels(["1E-7", "1E-6","1E-5","1E-4","1E-3","1E-2","1E-1"])

#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Error Bound')
plt.ylabel('PSNR')
#plt.ylabel('Decompression Bandwidth MB/s')
#line_plot.set(xscale="log", yscale="log")
#line_plot.ticklabel_format(useOffset=False, style='plain')
#plt.xlabel('Error Bound')
#plt.ylabel('ssim')
#plt.ylim(0, 300)
#line_plot.set(title='sz: CR')
bar_plot.set(title='SZ3: Cityscape PSNR')


#fig = line_plot.get_figure()
#bar_plot.figure.subplots_adjust(left = 0.16)

fig = bar_plot.get_figure()
#fig = plt.figure(figsize = (50, 50))

#heat_plot = sns.heatmap(data=df.loc[:,["CR"]], yticklabels=df.loc[:,["compressors"]])
#fig = heat_plot.get_figure()

for nr, p in enumerate(bar_plot.patches):

    # height of bar, which is basically the data value
    height = p.get_height() 

    # add text to specified position
    bar_plot.text(
        # bar to which data label will be added 
        # so this is the x-coordinate of the data label
        nr, 

        # height of data label: height / 2. is in the middle of the bar
        # so this is the y-coordinate of the data label
        height / 2., 

        # formatting of data label
        u'{:0.1f}'.format(height), 

        # color of data label
        color='black', 

        # size of data label
        fontsize=15, 

        # horizontal alignment: possible values are center, right, left
        ha='center', 

        # vertical alignment: possible values are top, bottom, center, baseline
        va='bottom'
    )


fig.savefig('rellis_base_RC.png')




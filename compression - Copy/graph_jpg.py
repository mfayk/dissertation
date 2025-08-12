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



df = pd.read_csv('rellis_base_lossy_jpg.csv')

xlab = "jpeg quality"
ylab = "PSNR"
title = "jpeg: psnr"

plt.figure(figsize=(30,5))

#bar_plot = sns.barplot(data=df, x="bound", y="CR", hue = "diff")
bar_plot = sns.barplot(data=df.query("compressor =='jpeg'"), x="quality", y="ssim")
#bar_plot.set_xticklabels(bar_plot.get_xticklabels(), fontsize=4.5)

#line_plot.axhline(0, color = 'r')
#plt.ylim(0, 75)

#bar_plot.set_xticklabels(["1E-7", "1E-6","1E-5","1E-4","1E-3","1E-2","1E-1"])

#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Quality')
plt.ylabel('ssim')
#plt.ylabel('Decompression Bandwidth MB/s')
#line_plot.set(xscale="log", yscale="log")
#line_plot.ticklabel_format(useOffset=False, style='plain')
#plt.xlabel('Error Bound')
#plt.ylabel('ssim')
#plt.ylim(0, 300)
#line_plot.set(title='sz: CR')
bar_plot.set(title='jpg: Cityscape ssim')


#fig = line_plot.get_figure()
#bar_plot.figure.subplots_adjust(left = 0.16)

fig = bar_plot.get_figure()
#fig = plt.figure(figsize = (50, 50))

#heat_plot = sns.heatmap(data=df.loc[:,["CR"]], yticklabels=df.loc[:,["compressors"]])
#fig = heat_plot.get_figure()
'''
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
'''

fig.savefig('rellis_base_jpeg.png')




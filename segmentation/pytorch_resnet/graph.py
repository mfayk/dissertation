#!/usr/bin/env python

from pathlib import Path
import json
#import libpressio
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
#from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import rawpy
import imageio

#from OCT_reader import get_OCTSpectralRawFrame

#from oct_converter.readers import FDS


from struct import unpack

import textwrap


    
    
#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']

#df = pd.read_csv('/scratch/mfaykus/seg_val/resnet_fine_test.csv')
df = pd.read_csv('/scratch/mfaykus/seg_val/resnet_fine_test.csv')
#df = pd.read_csv('data/bioFilm_tiff_lossy_threads.csv')

#diff0 = pd.read_csv('data/bioFilm_tiff_lossy_switch.csv')

'''
df = df[df['class'] != 'sidewalk']
df = df[df['class'] != 'building']
df = df[df['class'] != 'vegetation']
df = df[df['class'] != 'terrain']
df = df[df['class'] != 'sky']
df = df[df['class'] != 'person']
df = df[df['class'] != 'train']
df = df[df['class'] != 'car']
df = df[df['class'] != 'truck']
df = df[df['class'] != 'pole']
df = df[df['class'] != 'bus']
df = df[df['class'] != 'wall']
df = df[df['class'] != 'motorcycle']
df = df[df['class'] != 'bicycle']
df = df[df['class'] != 'traffic light']
df = df[df['class'] != 'truck']
df = df[df['class'] != 'fence']
df = df[df['class'] != 'traffic sign']
df = df[df['class'] != 'rider']
df = df[df['class'] != 'unlabeled']
'''
#1'truck', 1'bus', 1'train',1'motorcycle',1'rider',1'traffic light',

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
#difference =  ((df.loc[:,["CR"]] - diff0.loc[:,["CR"]])/diff0.loc[:,["CR"]])*100
#difference =  ((df.loc[df['filename'] == name,["CR"]] - diff0.loc[diff0['filename'] == name,["CR"]])/diff0.loc[diff0['filename'] == name,["CR"]])*100
#print("difference")
#print(difference.mean())


#print(df.groupby(['compressor'])['CR'].max())

#print(df.groupby(['diff'])['CR'].max())

#print(df.groupby(['compressor'])['ssim'].max())

#print(df.groupby(['diff'])['ssim'].max())

#print(df.groupby(['diff'])['CR'])

#pd.set_option("display.max_rows", 10000)
#pd.set_option("display.expand_frame_repr", True)
#pd.set_option('display.width', 10000)

#print(df.loc[(df['CR'] <= 1 ),["diff"]])
#print(df.loc[(df['CR'] <= 1 ),["filename"]])

#sns.set(rc={"figure.figsize":(14, 8)})

line_plot = sns.lineplot(data=df, x="error_bound", y="miou", hue = "model_num")
#line_plot = sns.lineplot(data=df.query("compressor =='zfp'"), x="bound", y="cBW", hue = "diff")
#line_plot.set(xlabel='bound', ylabel='Compression Ratio')
#line_plot.legend([],[], frameon=False)
#line_plot.plot(legend=False)
#line_plot.set(xlabel='bound', ylabel='Compression Ratio')
xlab = "Error Bound"
ylab = "miou"
title = "SZ3 resnet iou scores"
#sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))



#bar_plot = sns.barplot(data=df, x="error_bound", y="iou", hue = "class")
#bar_plot = sns.barplot(data=df.query("compressor =='sz'"), x="error_bound", y="CR", hue = "diff")
#bar_plot.set_xticklabels(bar_plot.get_xticklabels(), fontsize=4.5)

#line_plot.axhline(0, color = 'r')
#plt.ylim(0, 75)

#line_plot.set_xticklabels(["base", "1E-7", "1E-6","1E-5","1E-4","1E-3","1E-2","1E-1"])

#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Error Bound')
plt.ylabel('miou')
#plt.ylabel('Decompression Bandwidth MB/s')
#line_plot.set(xscale="log", yscale="log")
#line_plot.ticklabel_format(useOffset=False, style='plain')
#plt.xlabel('Error Bound')
#plt.ylabel('ssim')
#plt.ylim(0, 300)
line_plot.set(title='SZ3: Resnet iou scores')
#bar_plot.set(title='SZ: Pre-processing Compression Ratio (CR)')


fig = line_plot.get_figure()
#bar_plot.figure.subplots_adjust(left = 0.16)

#fig = bar_plot.get_figure()
#fig = plt.figure(figsize = (50, 50))

#heat_plot = sns.heatmap(data=df.loc[:,["CR"]], yticklabels=df.loc[:,["compressors"]])
#fig = heat_plot.get_figure()




fig.savefig('val_graph.png')




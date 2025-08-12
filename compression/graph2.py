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

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']

df = pd.read_csv('data/bioFilm_tiff_lossless.csv')

diff0 = pd.read_csv('data/bioFilm_tiff_lossless.csv')


print(df.groupby(['compressor'])['CR'].max())

print(df.groupby(['diff'])['CR'].max())

print(df.groupby(['compressor'])['dBW'].max())

print(df.groupby(['diff'])['dBW'].max())

print(df.groupby(['diff'])['CR'])


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


#line_plot = sns.lineplot(data=df.loc[df['filename'] == name], x="CL", y="CR", hue = "compressor")
#line_plot = sns.lineplot(data=df, x="CL", y="CR", hue = "compressor")
#line_plot.set(xlabel='Compression Level', ylabel='Compression Ratio')
#line_plot.set(xlabel='Compression Level', ylabel='Decompression Bandwidth: MB/s')


#bar_plot = sns.barplot(data=df.query("compressor =='zlib'"), x="CL", y="CR", hue = "diff")
bar_plot = sns.barplot(data=df, x="CL", y="CR", hue = "diff")

#bar_plot.set_xticklabels(bar_plot.get_xticklabels(), fontsize=4.5)

#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Compression Level')
plt.ylabel('Compression Ratio')
#plt.ylabel('Compression Bandwidth MB/s')
#plt.ylim(0, 4)
bar_plot.set(title='Lossless compressors: Compression Ratio (CR)')
#bar_plot.set(title='Lossless compressors: Compression Bandwidth')
#fig = line_plot.get_figure()
fig = bar_plot.get_figure()



#plt.yscale('log')
#plt.ylim(0, 4)
#fig = line_plot.get_figure()

fig.savefig('poster/tiff_lossless_thread.pdf')




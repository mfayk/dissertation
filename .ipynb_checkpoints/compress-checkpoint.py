#!/usr/bin/env python

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import math
import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from matplotlib import image
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
from sklearn import preprocessing
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


from oct_converter.readers import FDS
from struct import unpack

from skimage.metrics import structural_similarity

import time
import cv2

def compression(srcpath, paths, compressor, modes):
    #threads = [1,2,4,8,16,32]
    sizes = [505148*50, 481165*50, 321148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 509148*50, 509148*50, 509148*50, 509148*50]
    #sizes = [505148*50, 321148*50, 509148*50, 509148*50, 509148*50, 509148*50]
    #sizes = [505148*50, 481165*50, 509148*50, 509148*50, 509148*50, 509148*50]
    threads = [1]
    for thread in threads:
        mode_count = 0
        while mode_count < 3:
            mode = modes[mode_count]
            i = 0
            while i < len(paths):
                size = sizes[i]
                for comp in compressors:
                    j = 0
                    bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]

                    while(j<len(bounds)):
                        l=glob.glob(srcpath+paths[i])
                        l.sort()
                        #print(l)
                        counter = 0
                        for name in l:
                            counter = counter + 1
                            print(name)
                            
                            bound = bounds[j]

                            inp = image.imread(name)
                            input_data = np.asarray(inp)
                            
                            ori_data = input_data.copy()
                            print(input_data.shape)
                            
                            #exit(0)
                            input_data = input_data.astype(np.float32)


                            i_data = input_data.copy()
                            D_data = input_data.copy()
                            diff_data = input_data.copy()
                            decomp_data = input_data.copy()
                            de_data = input_data.copy()

                            #start timer
                            start = time.time()

                            input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)
                            #input_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min())
                            
                            normalize_time = time.time() - start                
                            diff_time = time.time() - start - normalize_time

                            compressor = libpressio.PressioCompressor.from_config({
                                    # configure which compressor to use
                                "compressor_id": comp,
                                    # configure the set of metrics to be gathered
                                    "early_config": {
                                        "pressio:metric": "composite",
                                        "pressio:nthreads":thread,
                                        "composite:plugins": ["time", "size", "error_stat"]
                                    },
                                    # configure SZ/zfp
                                    "compressor_config": {
                                        "pressio:abs": bound,
                                }})

                            comp_data = compressor.encode(input_data)

                            comp_time = time.time() - diff_time - normalize_time - start
                            encoding_time = time.time()- start

                            decomp_data = compressor.decode(comp_data, decomp_data)

                            D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255

                            de_comp_time = time.time() - encoding_time - start
                            end = time.time() - start

                            #D_data = decomp_data.copy()

                            print("The time of execution of above program is :",(end) * 10**3, "ms")
                            metrics = compressor.get_metrics()
                            D_data = D_data.astype(np.uint8)
                            (ssim, diff) = structural_similarity(i_data[:,:,0], D_data[:,:,0], data_range=255, full=True)
                            #diff = 0
                            #ssim = 0

                            diff = (diff * 255).astype("uint8")

                            print("SSIM: {}".format(ssim))
    #get size from libpressio

                            df.loc[len(df)] = [paths[i], bound, comp, ssim, metrics['error_stat:psnr'] , (size/encoding_time)/1000000, (size/de_comp_time)/1000000, size/metrics['size:compressed_size'],normalize_time,diff_time,comp_time,encoding_time,de_comp_time,end,mode,thread]



                            save = 0
                            if save == 0:
                                path = '/scratch/mfaykus/dissertation/datasets/rellis-images/compressed/train/'
                                im = Image.fromarray(D_data)
                                im.save(path + str(counter) + str(bound)  + '.jpg')
                                imageio.imwrite(path + str(counter) + str(bound)  + '.jpg', D_data)

                        j = j + 1
                i = i + 1
            mode_count = mode_count + 1
    return df




paths = ['train/rgb/*.jpg','test/rgb/*.jpg']
srcpath='/scratch/mfaykus/dissertation/datasets/rellis-images/'

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']
compressors = ['sz', 'zfp']

df = pd.DataFrame({
            'filename':[],
            'bound':[],
            'compressor':[],
            'ssim':[],
            'psnr':[],
            'cBW':[],
            'dBW':[],
            'CR':[],
            'normalize_time':[],
            'diff_time':[],
            'comp_time':[],
            'encoding_time':[],
            'de_comp_time':[],
            'total_time':[],
            'diff':[],
            'thread':[]})


#mode = ["Ldiff", "0diff", "base"]
mode = ["base"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv('compression/rellis_base_lossy.csv')

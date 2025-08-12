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
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re

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


from math import log10, sqrt 
import cv2 
import numpy as np

from skimage.metrics import structural_similarity
import cv2
import numpy as np
  
def PSNR(original, compressed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    mse = np.mean((original_gray - compressed_gray) ** 2)
    
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr 


def compression(srcpath, paths, compressor, modes):
    basepath='/scratch/mfaykus/dissertation/compression/cityscapes2/leftImg8bit/'
    path_base = ['val/*.png', 'train/*.png']
    threads = [1]
    for thread in threads:
        mode_count = [1]
        for modes in mode_count:
            mode = "base"   
            i = 0
            while i < len(paths):
                #print(len(paths))
                #size = sizes[i]

                for comp in compressors:
                    j = 1
                    
                    while(j<=100):
                        
                        
                        l=glob.glob(srcpath+ 'Q' + str(j) + '/leftImg8bit/' + paths[i])
                        l.sort()
                        
                        l1=glob.glob(basepath + path_base[i])
                        l1.sort()
                        
                        print(basepath + paths[i])
                        

                        counter = 0
                        for (name,name1) in zip(l,l1):
                            
                            print(name)
                            print(name1)

                            original = cv2.imread(name1) 
                            compressed = cv2.imread(name, 1)
               
                            psnr = PSNR(original, compressed)
                    
                            print(psnr)            

                            
                            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                            distorted_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

                            (score,diff) = structural_similarity(original_gray, distorted_gray, full=True)
                            
                            print(score)
                    
                            CR = os.path.getsize(name1) / os.path.getsize(name)
                
                            df.loc[len(df)] = [name, j, comp, score, psnr, CR]


                        j = j + 1
                        df.to_csv('rellis_base_lossy_jpg.csv')
                i = i + 1
            mode_count = mode_count[0] + 1
    return df




paths = ['val/*.jpg', 'train/*.jpg']

srcpath = '/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/jpeg/'

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']
compressors = ['jpeg']

df = pd.DataFrame({
            'filename':[],
            'quality':[],
            'compressor':[],
            'ssim':[],
            'psnr':[],
            'CR':[]})



mode = ["base"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv('rellis_base_lossy_jpg.csv')

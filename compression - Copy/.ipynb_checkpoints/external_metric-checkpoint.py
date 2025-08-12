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
import argparse
import numpy as np
import cv2
from math import log10, sqrt 
import itertools 


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 


base_path = "/zfs/fthpc/mfaykus/datasets/cityscapes2/leftImg8bit/"
base_id = "/zfs/fthpc/mfaykus/datasets/cityscapes2/gtFine/"
comp_path = "/zfs/fthpc/mfaykus/datasets/cityscapes2/compressed/"

paths = ['val/*.png']
paths_id = ['val/*_labelIds.png', 'train/*_labelIds.png']
bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]

bounds_name = ['1E-7','1E-6','1E-5','1E-4','1E-3','1E-2','1E-1']


classes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
psnr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

j=0
while(j<len(bounds_name)):
    i=0
    classes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    psnr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while i < len(paths):
        
        lb=glob.glob(base_path+paths[i])
        lb.sort()
    
        ID=glob.glob(base_id+paths_id[i])
        ID.sort()
        
        lc=glob.glob(comp_path + bounds_name[j] + '/leftImg8bit/' + paths[i])
        lc.sort()
        
        val_avg = []
        ssim_avg = []
        for name1, name2, name_id in zip(lb,lc,ID):
            
            base_img = cv2.imread(name1)
            comp_img = cv2.imread(name2)
            ID_img = cv2.imread(name_id)
            
            #print(type(ID_img))
            #print(np.unique(ID_img))
            
            #for elem in np.nditer(ID_img):
            #    classes[elem] += 1
            
            #print(name2)
            #print(classes)
            
            #val = PSNR(base_img, comp_img)
            #val_avg.append(val)
            #print(base_img.shape)
            for base_i, comp_i, id_i in zip(base_img, comp_img, ID_img):
                for base_j, comp_j, id_j in zip(base_i, comp_i, id_i):
                    #print(base_j.shape)
                    #print(base_j)
                    #print(comp_j)
                    #print(id_j[0])
                    val = PSNR(base_j, comp_j)
                    #print(val)
                    psnr[id_j[0]] += val
                    classes[id_j[0]] += 1
                    
                    #print(psnr)
                    #print(classes)
                    #print(psnr/classes)
                    
                    
                
            
            
        i = i + 1
    print(bounds_name[j])
    print(psnr)
    counter = 0
    for ps, cl in zip (psnr, classes):
        if(cl != 0):
            print(ps/cl)
            psnr[counter] = ps/cl
            counter += 1
                    
    print(psnr)
    print(classes)
        #print("PSNR AVG")
        #print(np.average(np.array(val_avg)))
        
    #print(psnr./classes)
  
    j = j + 1
    
print(classes)
norm = [float(i)/max(classes) for i in classes]
print(norm)
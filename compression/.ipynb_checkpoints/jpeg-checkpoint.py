#!/usr/bin/env python

from pathlib import Path
import json
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

import imageio
import shutil 

from struct import unpack

from skimage.metrics import structural_similarity

import time

from io import StringIO # "import StringIO" directly in python2
from PIL import Image
from io import BytesIO

paths = ['val/*.png', 'train/*.png', 'test/*.png']
paths = ['val/*.png']
srcpath='/scratch/mfaykus/dissertation/datasets/cityscapes2/leftImg8bit/'
srcpath='/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/leftImg8bit/'
paths = ['train/rgb/*.png']
#srcpath='/project/jonccal/fthpc/mfaykus/datasets/rellis-images/compressed/sz3/'
srcpath = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split-png/'


flag = 0
i = 0
while i < len(paths):

    l=glob.glob(srcpath+paths[i])
    l.sort()

    for name in l:
        print(name)
        im1 = Image.open(name)

        Quality = 1
        while Quality <= 100:
            buffer = BytesIO()
            im1.save(buffer, "JPEG", quality=Quality)
            
            dest = '/scratch/mfaykus/dissertation/datasets/rellis/compressed/jpeg/'
            '''
            if i ==0:
                dest_2 = '/leftImg8bit/val/'
            elif i == 1:
                dest_2 = '/leftImg8bit/train/'
            elif i == 2:
                dest_2 = '/leftImg8bit/test/'
            '''
            
            dest_2 = '/leftImg8bit/val/'
            
            path = dest + 'Q' + str(Quality) + dest_2

            
            # make the result directory
            if not os.path.exists(path):
                os.makedirs(path)
            
            filename = os.path.basename(name).split('/')[-1]
            
            filename = filename.split('.', 1)[0]
            filename = str(filename) + '.jpg'
            location = path + str(filename)
            
            print(location)
            
            
                
            with open(location, "wb") as handle:  # Open in binary mode
                handle.write(buffer.getvalue())
            '''    

            destination_dir = (dest + 'Q' + str(Quality)+'/gtFine')
            os.makedirs(destination_dir, exist_ok=True)

            copy_gt = '/scratch/mfaykus/dissertation/datasets/rellis/gtFine'
'''
            if flag == 5 and Quality > 95:
                shutil.copytree(copy_gt, destination_dir, dirs_exist_ok=True) 


            Quality += 1
        flag = 1
    i += 1
            
            
            

    






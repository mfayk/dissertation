#!/usr/bin/env python


#evaluate 1e3

print("here")

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import math
import numpy as np
import glob
from matplotlib import image
import matplotlib.pyplot as plt
#from skimage.metrics import structural_similarity as ssim
#import cv2
#import pandas as pd



def compression(srcpath, paths, compressor, modes):
    threads = [1]
    for thread in threads:
        print("here1")
        mode_count = [1]
        for modes in mode_count:
            print("here2")
            mode = "base"
            
            i = 0
            while i < len(paths):
                print("here3")
                print(len(paths))
                #size = sizes[i]
                for comp in compressors:
                    print("here4")
                    j = 0
                    bounds = [1e-3,math.exp(-2)]
                    print(math.exp(-3))
                    print(math.pow(1,-3))
                    print(1e-3)
                    exit()

                    while(j<len(bounds)):
                        print("here5")
                        l=glob.glob(srcpath+paths[i])
                        l.sort()
                        #print(l)
                        counter = 0
                        for name in l:
                            print("here6")
                            print(name)
                            counter = counter + 1
                            
                            bound = bounds[j]

                            print(name)
                            inp = image.imread(name)
                            input_data = np.asarray(inp)
                            print(input_data.dtype)
                            exit()
                            image_np_array = np.array(inp)
                            
                            if inp.dtype != np.uint8:
                                inp = (inp * 255).astype(np.uint8)
                                
                            plt.imsave('inps_array_img.png', inp)
                            
                            print("before normalization")
                            print(inp[:,:,0])
                            print(inp[:,:,0].shape)
                            np.savetxt("before_norm.csv", inp[:,:,0], delimiter=",")
                            #exit()
                            ori_data = input_data.copy()
                            #print(input_data.shape)
                            
                            #input_data = input_data.astype(np.float32)


                            i_data = input_data.copy()
                            D_data = input_data.copy()
                            diff_data = input_data.copy()
                            decomp_data = input_data.copy()
                            de_data = input_data.copy()

                            
                           
                            #pd.DataFrame(ori_data.reshape(ori_data.shape[0], -1)).to_csv('before_norm.csv', index=False, header=False)

                            #input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32f)
                            #input_data = (input_data-np.min(input_data))/(np.max(input_data)-np.min(input_data))
                            #input_data = (input_data - input_data.min())/ (input_data.max() - input_data.min())
                            print("after normilization")
                            print(input_data)
                            np.savetxt("after_norm.csv", input_data[:,:,0], delimiter=",")

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

                            decomp_data = compressor.decode(comp_data, decomp_data)

                            print("decompressed before normazation")
                            print(decomp_data[:,:,0])
                            np.savetxt("decomp_before_norm.csv", decomp_data[:,:,0], delimiter=",")
                            #D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255
                            
                            D_data = (decomp_data * 255).astype(np.uint8)
                            print("decompressed after normalziation")
                            print(D_data[:,:,0])
                            np.savetxt("decomp_after_norm.csv", D_data[:,:,0], delimiter=",")
                            print(len(D_data))
                            

                            plt.imsave('decomp_array_img.png', D_data)
                            
                            
                            mse = np.mean((input_data - decomp_data) ** 2)

                            # Calculate PSNR
                            psnr = 10 * np.log10(1 / mse)  # Since image is in [0, 1] range, MAX = 1

                            # Calculate SSIM
                           # ssim_value = ssim(input_data, decomp_data, multichannel=True)

                            print(f"MSE: {mse}")
                            print(f"PSNR: {psnr}")
                           # print(f"SSIM: {ssim_value}")
                            
                            exit()


                        j = j + 1
                i = i + 1
            mode_count = mode_count[0] + 1
    return 0




paths = [ 'train/*.png', 'val/*.png']
srcpath='/project/jonccal/fthpc/mfaykus/datasets/cityscapes2/leftImg8bit/'

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']
compressors = ['sz3']


mode = ["base"]

print("here")

data = compression(srcpath, paths, compressors, mode)

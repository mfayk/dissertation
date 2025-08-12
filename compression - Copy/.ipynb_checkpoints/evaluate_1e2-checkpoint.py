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
#from skimage.metrics import structural_similarity as ssim
#import cv2
#import pandas as pd
from PIL import Image
import imageio


def compression(srcpath, paths, compressor, modes):
    threads = [1]
    for thread in threads:
        #print("here1")
        mode_count = [1]
        for modes in mode_count:
            #print("here2")
            mode = "base"
            
            i = 1
            while i < len(paths):
                #print("here3")
                #print(len(paths))
                #size = sizes[i]
                for comp in compressors:
                    #print("here4")
                    j = 3
                    #bounds = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
                    bounds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12]
                    

                    while(j<len(bounds)):
                        #print("here5")
                        l=glob.glob(srcpath+paths[i])
                        l.sort()
                        #print(l)
                        counter = 0
                        for name in l:
                            #print("here6")
                            #print(name)
                            counter = counter + 1
                            
                            bound = bounds[j]

                            print(name)
                            inp = image.imread(name)
                            input_data = np.asarray(inp)
                            
                            image_np_array = np.array(inp)
                            
                            if inp.dtype != np.uint8:
                                inp = (inp * 255).astype(np.uint8)
                                
                            #plt.imsave('inps_array_img2.png', inp)
                            
                            #print("before normalization")
                            #print(inp[:,:,0])
                            #print(inp[:,:,0].shape)
                            #np.savetxt("before_norm2.csv", inp[:,:,0], delimiter=",")
                            #exit()
                            ori_data = input_data.copy()
                            #print(input_data.shape)
                            
                            input_data = input_data.astype(np.float32)


                            i_data = input_data.copy()
                            D_data = input_data.copy()
                            diff_data = input_data.copy()
                            decomp_data = input_data.copy()
                            de_data = input_data.copy()

                            
                           
                            #pd.DataFrame(ori_data.reshape(ori_data.shape[0], -1)).to_csv('before_norm.csv', index=False, header=False)

                            #input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32f)
                            #input_data = (input_data-np.min(input_data))/(np.max(input_data)-np.min(input_data))
                            #input_data = (input_data - input_data.min())/ (input_data.max() - input_data.min())
                            #print("after normilization")
                            #print(input_data)
                            #np.savetxt("after_norm2.csv", input_data[:,:,0], delimiter=",")

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

                            #print("decompressed before normazation")
                            #print(decomp_data[:,:,0])
                            #np.savetxt("decomp_before_norm2.csv", decomp_data[:,:,0], delimiter=",")
                            #D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255
                            
                            D_data = (decomp_data * 255).astype(np.uint8)
                            #print("decompressed after normalziation")
                            #print(D_data[:,:,0])
                            #np.savetxt("decomp_after_norm2.csv", D_data[:,:,0], delimiter=",")
                            #print(len(D_data))
                            

                            #plt.imsave('decomp_array_img2.png', D_data)
                            
                            metrics = compressor.get_metrics()
                            mse = np.mean((input_data - decomp_data) ** 2)

                            # Calculate PSNR
                            psnr = 10 * np.log10(1 / mse)  # Since image is in [0, 1] range, MAX = 1

                            # Calculate SSIM
                            #ssim_value = ssim(input_data, decomp_data, multichannel=True)

                            #print(f"MSE: {mse}")
                            #print(f"PSNR: {psnr}")
                           # print(f"SSIM: {ssim_value}")
                        
                            psnr_values = []
                            channel_names = ['Red', 'Green', 'Blue']

                            #for i in range(3):
                            #    channel_mse = np.mean((input_data[:,:,i] - decomp_data[:,:,i]) ** 2)
                            #    channel_psnr = 10 * np.log10(1 / channel_mse)
                            #    psnr_values.append(channel_psnr)
                            #    print(f"{channel_names[i]} channel PSNR: {channel_psnr:.2f} dB")

                            # Find the channel with highest PSNR
                            #max_index = np.argmax(psnr_values)
                            #print(f"\nHighest PSNR is in the {channel_names[max_index]} channel: {psnr_values[max_index]:.2f} dB")


                            size = os.path.getsize(name)

                            #df.loc[len(df)] = [paths[i], bound, comp,  metrics['error_stat:psnr'] , size/metrics['size:compressed_size'],mode,thread]


                            print(name)
                            filename = os.path.basename(name).split('/')[-1]
                            save = 0
                            if save == 0:
                                dest = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/sz3_finetuned_png/'
                                
                                if i ==0:
                                    dest_2 = '/train/rgb/'
                                elif i == 1:
                                    dest_2 = '/test/rgb/'
                                                             
                                        
                                #0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12
                                if(j == 0):
                                    path = dest + '05' + dest_2
                                if(j == 1):
                                    path = dest + '06' + dest_2
                                if(j == 2):
                                    path = dest + '08' + dest_2
                                if(j == 3):
                                    path = dest + '09' + dest_2
                                if(j == 4):
                                    path = dest + '11' + dest_2
                                if(j == 5):
                                    path = dest + '12' + dest_2
                                if(j == 6):
                                    path = dest + '1E-1' + dest_2

                                # make the result directory
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    
                                im = Image.fromarray(D_data)
                                
                                print(path + str(filename))
                                
                                im.save(path + str(filename))
                                imageio.imwrite(path + str(filename), D_data)

                        j = j + 1
                i = i + 1
            mode_count = mode_count[0] + 1
    return 0




paths = ['train/rgb/*.png', 'test/rgb/*.png']
#srcpath='/project/jonccal/fthpc/mfaykus/datasets/rellis-images/compressed/sz3/'
srcpath = '/scratch/mfaykus/dissertation/datasets/Rellis-3D/Rellis-3D-camera-split-png/'
#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']
compressors = ['sz3']


mode = ["base"]

print("here")

data = compression(srcpath, paths, compressors, mode)

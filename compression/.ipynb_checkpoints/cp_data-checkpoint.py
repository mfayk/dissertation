import shutil 

level = 1



while level < 101:

    src = '/scratch/mfaykus/dissertation/datasets/cityscapes2/gt/'


    dst = '/scratch/mfaykus/dissertation/datasets/cityscapes2/compressed/jpeg/Q' + str(level) + '/'

    shutil.copytree(src, dst, dirs_exist_ok=True)
    
    level += 1

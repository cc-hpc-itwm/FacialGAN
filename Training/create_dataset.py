import os, glob
import pandas as pd
from shutil import copyfile

path =  "./data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
dst = "./data/celeba_hq/anno/"

# We first join all the mask under the folder
for i in range(15):
    path_tmp = os.path.join(path,str(i)) 
    for filepath in glob.glob(path_tmp+'/*.png'):
        name = filepath.split("/")
        filepath2 = os.path.join(dst,name[-1]) 
        copyfile(filepath, filepath2)   
        
        
# We align the names of images with their annotation mask
filename = "./data/anno.txt"
df = pd.read_fwf(filename)

folder_name = ["./data/celeba_hq/train/male/","./data/celeba_hq/train/female/",
               "./data/celeba_hq/val/male/","./data/celeba_hq/val/female/"]

for i in range(len(folder_name)):
    for filepath in glob.glob(folder_name[i]+"*.jpg"):
        file = filepath.split("/")
        res_row = df.loc[df['orig_file'] == file[-1]]
        res_idx = res_row.index[0]
        refile = str(res_idx).zfill(5) +".jpg"  
        filepath2 = os.path.join(folder_name[i],refile)
        os.rename(filepath, filepath2)


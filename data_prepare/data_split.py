"""import os
img_path = "/home/luosy/data3382/image/"
mask_path =  "/home/luosy/data3382/mask/"
val_img_path =  "/home/luosy/data3382/val_image/"
val_mask_path =  "/home/luosy/data3382/val_mask/"
index = [ 1,3,4, 6, 9, 15, 18, 21, 24, 27, 30, 33, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72,74, 75, 78, 81, 84, 87, 90, 93, 95,96, 99, 109, 113, 117, 121, 125, 129,130]
val_index = [3,8,12,14,18,25,64,67,85,97,99,101,105,113,117,118,121,125]  # 15
# 75
for idx in val_index:
    cmd_dd1 = "mv /home/luosy/data3382/image/{}_* /home/luosy/data3382/val_image/".format(idx)
    cmd_dd2 = "mv /home/luosy/data3382/mask/{}_* /home/luosy/data3382/val_mask/".format(idx)
    os.system(cmd_dd1)
    os.system(cmd_dd2)"""

import os
import shutil
img_path = "U:/paper3testimg/test_image"
mask_path =  "U:/paper3testimg/test_mask"
val_img_path =  "U:/paper3testimg/val_img"
val_mask_path =  "U:/paper3testimg/val_mask"
index = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 101, 105, 109, 113, 117, 121, 125, 129]
val_index = [3,8,12,14,18,25,64,67,85,97,99,101,105,113,117,118,121,125]
imgfiles = os.listdir(img_path)
imgfiles_path = [os.path.join(img_path,idx) for idx in imgfiles]
maskfiles = os.listdir(mask_path)
maskfiles_path = [os.path.join(mask_path,idx) for idx in maskfiles]
for idx in range(len(imgfiles_path)):
    name = int(imgfiles[idx].split('_')[0])
    if name in val_index:
        shutil.move(imgfiles_path[idx],val_img_path)

for idx in range(len(maskfiles_path)):
    name = int(maskfiles[idx].split('_')[0])
    if name in val_index:
        print(name)
        shutil.move(maskfiles_path[idx],val_mask_path)



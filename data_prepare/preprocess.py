
indexpp = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 101, 105, 109, 113, 117, 121, 125, 129]
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import scipy.ndimage as ndi
np.random.seed(137)
ct_name = ".nii"
mask_name = ".nii"
np.random.seed(137)
ct_path = 'C:/Users/luosy/Desktop/lits_res/rawdata/CT'
seg_path = 'C:/Users/luosy/Desktop/lits_res/rawdata/SEG'
png_path = './png/'

outputImg_path = "U:/liverdata256/image"
outputMask_path = "U:/liverdata256/mask"

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 3
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i, :, :]
        if np.sum(img_slice_begin) > 0:
            # bb[0] = np.max([i - bb_extend, 0])
            bb[0] = 2*i//3

            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0] - 1 - i, :, :]
        if np.sum(img_slice_end) > 0:
            # bb[1] = np.min([img_shape[0] - 1 - i + bb_extend, img_shape[0] - 1])
            bb[1] = img_shape[0] - 1 - i +( 1 + i )//3

            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:, i, :]
        if np.sum(img_slice_begin) > 0:
            bb[2] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:, img_shape[1] - 1 - i, :]
        if np.sum(img_slice_end) > 0:
            bb[3] = np.min([img_shape[1] - 1 - i + bb_extend, img_shape[1] - 1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:, :, i]
        if np.sum(img_slice_begin) > 0:
            bb[4] = np.max([i - bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:, :, img_shape[2] - 1 - i]
        if np.sum(img_slice_end) > 0:
            bb[5] = np.min([img_shape[2] - 1 - i + bb_extend, img_shape[2] - 1])
            break

    return bb
def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]


if __name__ == "__main__":

    for index, file in enumerate(tqdm(os.listdir(ct_path))):

        # 获取CT图像及Mask数据
        name = (file.split('.')[0]).split('-')[-1]
        if  int(name)  in indexpp:
            print("name = ", name)
            continue
        else:
            print("{} not in index pp".format(name))
        # print("name = ", name)
        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)

        # mask_array[mask_array == 1] = 0  # 肿瘤
        # mask_array[mask_array == 2] = 1

        # 阈值截取
        ct_array[ct_array > 240] = 240
        ct_array[ct_array < -100] = -100

        ct_array = ct_array.astype(np.float32)
        ct_array = (ct_array+100.)/340.

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        # z = np.any(mask_array, axis=(1, 2))
        # start_slice, end_slice = np.where(z)[0][[0, -1]]

        bb = find_bb(mask_array)
        print("bb = ",bb)
        s1 = abs(bb[3]-bb[2])
        s2 = abs(bb[5]-bb[4])
        off = np.random.randint(10,20)
        print("off = ",off)
        if s1 > s2:
            dp = (s1 - s2)//2
            dp = min(dp+off,bb[4])
            print("dp = ", dp)
            bb[4] = bb[4] - dp
            bb[5] = min(bb[4] + s1 + 2*off,512)
            bb[2] = max(bb[2]-off,0)
            bb[3] = min(bb[2]+s1 + 2*off,512)
        else:
            dp =  (s2-s1) // 2
            dp = min(dp+off,bb[2])
            print("dp = ", dp)
            bb[2] = bb[2] - dp
            bb[3] = min(bb[2] + s2 + 2*off,512)
            bb[4] = max(0,bb[4]-off)
            bb[5] = min(bb[4]+s2 + 2*off,512)
        # print("bb modify = ", bb)
        # seg_array = mask_array[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        # print('effective shape:', seg_array.shape)
        # print('cs', start_slice, end_slice, file)
        # continue
        start_slice = max(0, bb[0] - 1)
        end_slice = min(mask_array.shape[0] - 1, bb[1] + 2)

        ct_crop = ct_array[start_slice:end_slice,bb[2]:bb[3],bb[4]:bb[5]]
        mask_crop = mask_array[start_slice:end_slice,bb[2]:bb[3],bb[4]:bb[5]]


        # ct_crop = ct_crop[:, yl:yr,xl:xr]
        # mask_crop = mask_crop[:, yl:yr,xl:xr]
        ct_crop = ndi.zoom(ct_crop, (1,256 / ct_crop.shape[1], 256 / ct_crop.shape[2]), order=3)
        mask_crop = ndi.zoom(mask_crop, (1,256 / mask_crop.shape[1], 256 / mask_crop.shape[2]), order=0)

        print("ct_crop shape = ",ct_crop.shape)
        print("mask_crop shape = ", mask_crop.shape)
        # 切片处理,并去掉没有病灶的切片
        if int(np.sum(mask_crop)) != 0:
            for n_slice in range(mask_crop.shape[0]):
                maskImg = mask_crop[n_slice, :, :]
                ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 1), np.float)
                ctImageArray[:, :, 0] = ct_crop[n_slice, :, :]
                imagepath = outputImg_path + "/" + str(name) + "_" + str(n_slice) + ".npy"
                maskpath = outputMask_path + "/" + str(name) + "_" + str(n_slice) + ".npy"

                np.save(imagepath, ctImageArray)  # (448，448,3) np.float dtype('float64')
                np.save(maskpath, maskImg)  # (448，448) dtype('uint8') 值为0 1 2
        else:
            continue
    print("Done！")
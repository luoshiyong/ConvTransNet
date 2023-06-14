from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label
import skimage.measure as measure
import skimage.morphology as morphology
import copy


indir = "U:/LITS_2017_TEST_DATA/test5seg/EEDiff"
outdir = "U:/LITS_2017_TEST_DATA/EEDiff_out"


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii', join=False)
    print(niftis)
    print('hello world')
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        # print("n = {} | identifier = {}".format(n,identifier))
        # outfname = join(outdir, "%s.nii" % identifier)
        outfname = join(outdir, n)
        # print('out',outfname)
        # return
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        img_tumor = img_npy.copy()
        img_npy[img_npy > 0] = 1
        pred_seg = img_npy.astype(np.uint8)
        img_new = sitk.GetImageFromArray(pred_seg)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    export_segmentations_postprocess(indir, outdir)
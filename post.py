from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label
import skimage.measure as measure
import skimage.morphology as morphology
import copy


"""indir = "U:/paper3/pred/SANet_cfm_53"
outdir = "U:/paper3/pred/SANet31_deformable_post"""

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
        """
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())

        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        """
        pred_seg = img_npy.astype(np.uint8)
        liver_seg = copy.deepcopy(pred_seg)
        liver_seg = measure.label(liver_seg, 4)
        props = measure.regionprops(liver_seg)

        max_area = 0
        max_index = 0
        for index, prop in enumerate(props, start=1):
            if prop.area > max_area:
                max_area = prop.area
                max_index = index

        liver_seg[liver_seg != max_index] = 0
        liver_seg[liver_seg == max_index] = 1

        liver_seg = liver_seg.astype(np.bool)
        morphology.remove_small_holes(liver_seg, 4e4, connectivity=2, in_place=True)
        img_npy = liver_seg.astype(np.uint8)
        # liver_seg = liver_seg.astype(np.bool)
        # morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
        # liver_seg = liver_seg.astype(np.uint8)

        img_npy[(img_tumor == 2) * (img_npy == 1)] = 2
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    export_segmentations_postprocess(indir, outdir)

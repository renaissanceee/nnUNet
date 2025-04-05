import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
def adjust_range(data):
    return (data * 255).astype(np.uint8)

def compute_uncertainty(prob_slice):
    eps = 1e-6  # avoid log(0) for NaN
    # H(p)=−plogp−(1−p)log(1−p)
    return -prob_slice * np.log(prob_slice + eps) - (1 - prob_slice) * np.log(1 - prob_slice + eps)

def visualize_middle_slice(file, suffix, save_path):
    img = nib.load(file + suffix)
    data = img.get_fdata()
    slice_idx = data.shape[2] // 3  # middle z-axis
    slice_data = data[:, :, slice_idx]
    ## 0,1,2,3
    colors = ["black", "green", "red", "blue"];cmap = ListedColormap(colors)
    # import pdb;pdb.set_trace()
    # print(np.sum(slice_data== 2),np.sum(slice_data== 1)+np.sum(slice_data== 2)+np.sum(slice_data== 3))
    plt.imsave(save_path+f"_{slice_idx}.png", slice_data, cmap=cmap)

def visualize_prob_middle_slice(file, suffix, save_path):
    img = np.load(file + suffix)['probabilities']
    data = np.transpose(img, (3, 2, 1, 0)) # (3,155,240,240)->(240,240,155,3)
    slice_idx = data.shape[2] // 3  # middle z-axis
    max_prob_slice = np.max(data, axis=-1)[:, :, slice_idx]
    nec_prob_slice = data[:, :, slice_idx,2]
    wt_prob_slice = data[:, :, slice_idx,1]+data[:, :, slice_idx,2]+data[:, :, slice_idx,3]
    # entropy-enc
    nec_uncertainty = compute_uncertainty(nec_prob_slice)
    wt_uncertainty = compute_uncertainty(wt_prob_slice)
    plt.imsave(save_path+f"_{slice_idx}_max.png", max_prob_slice, cmap="jet")# 彩色：越暖越大, 蓝-黄绿-红
    # plt.imsave(save_path + f"_{slice_idx}_nec.png", nec_prob_slice, cmap="jet")
    # plt.imsave(save_path + f"_{slice_idx}_wt.png", wt_prob_slice, cmap="jet")
    plt.imsave(save_path + f"_{slice_idx}_nec.png", nec_uncertainty, cmap="jet")
    plt.imsave(save_path + f"_{slice_idx}_wt.png", wt_uncertainty, cmap="jet")

# file_name = "BraTS2021_00000"
# file_name = "BraTS2021_00009"
# file_name = "BraTS2021_00016"
# file_name = "BraTS2021_00024"
file_name = "BraTS2021_00028"
# file_name = "BraTS2021_00031"
# file_name = "BraTS2021_00035"
# file_name = "BraTS2021_00045"
## pred_seg ##
seg_pred_root = "./nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation/"
seg_save_root = "./nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/val_seg/"
os.makedirs(seg_save_root, exist_ok=True)
visualize_middle_slice(os.path.join(seg_pred_root,file_name), ".nii.gz", os.path.join(seg_save_root,file_name))
## gt_seg ##
gt_file_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/"
gt_save_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/val_gt"
os.makedirs(gt_save_root, exist_ok=True)
visualize_middle_slice(os.path.join(gt_file_root,file_name), ".nii.gz", os.path.join(gt_save_root,file_name))
## pred_prob ##
prob_pred_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/"
prob_save_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/val_prob"
os.makedirs(prob_save_root, exist_ok=True)
visualize_prob_middle_slice(os.path.join(prob_pred_root,file_name), ".npz", os.path.join(prob_save_root,file_name))



file_name = "BraTS2021_00031"

## pred_seg ##
seg_pred_root = "./nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation/"
seg_save_root = "./nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/val_seg/"
os.makedirs(seg_save_root, exist_ok=True)
visualize_middle_slice(os.path.join(seg_pred_root,file_name), ".nii.gz", os.path.join(seg_save_root,file_name))
## gt_seg ##
gt_file_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/"
gt_save_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/val_gt"
os.makedirs(gt_save_root, exist_ok=True)
visualize_middle_slice(os.path.join(gt_file_root,file_name), ".nii.gz", os.path.join(gt_save_root,file_name))
## pred_prob ##
prob_pred_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/"
prob_save_root = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/val_prob"
os.makedirs(prob_save_root, exist_ok=True)
visualize_prob_middle_slice(os.path.join(prob_pred_root,file_name), ".npz", os.path.join(prob_save_root,file_name))
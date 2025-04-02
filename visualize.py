import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def visualize_middle_slice(file, suffix, save_path):
    img = nib.load(file + suffix)
    data = img.get_fdata()
    slice_idx = data.shape[2] // 2  # middle z-axis
    slice_data = data[:, :, slice_idx]
    ## 0-255
    # slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    # slice_data = slice_data.astype(np.uint8)
    # plt.imsave(save_path, slice_data, cmap="gray")
    # plt.imsave(save_path, slice_data, cmap="hot")
    # plt.imsave(save_path, slice_data, cmap="jet")

    ## 0,1,2,3
    colors = ["black", "green", "red", "blue"];cmap = ListedColormap(colors)
    plt.imsave(save_path+f"_{slice_idx}.png", slice_data, cmap=cmap)

def visualize_prob_middle_slice(file, suffix, save_path):
    img = np.load(file + suffix)['probabilities']
    data = np.transpose(img, (3, 2, 1, 0)) # # (3,155,240,240)->(240,240,155,3)
    data = np.max(data, axis=-1)
    slice_idx = data.shape[2] // 2  # middle z-axis
    slice_data = data[:, :, slice_idx]
    # colors = ["black", "green", "red", "blue"];cmap = ListedColormap(colors)
    # plt.imsave(save_path+f"_{slice_idx}.png", slice_data, cmap=cmap)
    ## 0-255
    slice_data = slice_data * 255
    slice_data = slice_data.astype(np.uint8)
    plt.imsave(save_path+f"_{slice_idx}_gray.png", slice_data, cmap="gray")
    plt.imsave(save_path+f"_{slice_idx}_hot.png", slice_data, cmap="hot")# 热力图：越红越大, 黑-白
    plt.imsave(save_path+f"_{slice_idx}_jet.png", slice_data, cmap="jet")# 彩色：越暖越大, 蓝-黄绿-红

file_name = "BraTS2021_00000"
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

# npz_path = os.path.join(preds_root, file_name+'.npz')
# probs = np.load(npz_path)
# probs = probs['probabilities']# (3,155,240,240)

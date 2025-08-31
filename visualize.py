import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
def adjust_range(data):
    return (data * 255).astype(np.uint8)
slice_idx = 20
def compute_uncertainty(prob_slice):# H(p)=−plogp−(1−p)log(1−p)
    eps = 1e-6
    return -prob_slice * np.log(prob_slice + eps) - (1 - prob_slice) * np.log(1 - prob_slice + eps)

def visualize_middle_slice(file, suffix, save_path):
    img = nib.load(file + suffix)
    data = img.get_fdata()
    # slice_idx = data.shape[2] // 3  # middle z-axis
    # slice_idx = data.shape[2] // 3 *2  # middle z-axis
    slice_data = data[:, :, slice_idx]
    ## 0,1,2,3
    colors = ["black", "green", "red", "blue"];cmap = ListedColormap(colors)
    plt.imsave(save_path+f"_{slice_idx}.png", slice_data, cmap=cmap)

def visualize_prob_middle_slice(file, suffix, save_path):
    img = np.load(file + suffix)['probabilities']
    data = np.transpose(img, (3, 2, 1, 0)) # (4,155,240,240)->(240,240,155,4)
    # slice_idx = data.shape[2] // 3  # middle z-axis
    max_prob_slice = np.max(data, axis=-1)[:, :, slice_idx]
    nec_prob_slice = data[:, :, slice_idx,2]
    wt_prob_slice = data[:, :, slice_idx,1]+data[:, :, slice_idx,2]+data[:, :, slice_idx,3]
    # entropy-enc
    nec_uncertainty = compute_uncertainty(nec_prob_slice)
    wt_uncertainty = compute_uncertainty(wt_prob_slice)
    plt.imsave(save_path+f"_{slice_idx}_max.png", max_prob_slice, cmap="jet")# 彩色：越暖越大, 蓝-黄绿-红

    plt.imsave(save_path + f"_{slice_idx}_nec.png", nec_uncertainty, cmap="jet")
    plt.imsave(save_path + f"_{slice_idx}_wt.png", wt_uncertainty, cmap="jet")

def visualize_input_middle_slice(input_root, file_name, suffix, save_path, slice_idx):
    file = os.path.join(input_root, f'{file_name}_0000')
    # 加载第一张图
    img1 = nib.load(file + suffix)
    data1 = img1.get_fdata()  # shape: (H, W, D) or (C, H, W, D)
    slice_data1 = data1[:, :, slice_idx]
    # 加载第二张图
    img2 = nib.load(
        f'nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerDiceLoss__nnUNetPlans__3d_fullres/fold_0/test/{file_name}{suffix}')
    data2 = img2.get_fdata()
    slice_data2 = data2[:, :, slice_idx]

    # 设置第二张图的颜色映射
    colors = ["black", "green", "red", "blue"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots()
    
    # ax.imshow(slice_data1, cmap='gray')  # 显示第一张图
    # ax.imshow(slice_data2, cmap=cmap, alpha=0.8)  # 显示第二张图并设置透明度

    ax.imshow(slice_data1, cmap='gray', alpha=1)  # 显示第一张图，完全不透明
    slice_data2_with_nan = np.where(slice_data2 == 0, np.nan, slice_data2)
    ax.imshow(slice_data2_with_nan, cmap=cmap, alpha=0.5, vmin=0, vmax=np.max(slice_data2))

    # 保存叠加后的图像
    ax.axis('off')
    plt.savefig(save_path + f"_{slice_idx}_overlay.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# input
input_root = "/leonardo_scratch/fast/EUHPC_B26_036/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/imagesTs/fold_0/"
input_save_root = "./vis_input"
os.makedirs(input_save_root, exist_ok=True)

# slice_idx = 155 // 3 *2 # middle z-axis
file_name = "BraTS2021_00009"
slice_idx = 77
visualize_input_middle_slice(input_root,file_name, ".nii.gz", os.path.join(input_save_root,file_name),slice_idx)
file_name = "BraTS2021_00031"
slice_idx = 51
visualize_input_middle_slice(input_root,file_name, ".nii.gz", os.path.join(input_save_root,file_name),slice_idx)

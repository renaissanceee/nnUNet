import torch
import numpy as np
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from typing import Tuple, List, Union, Optional, Any
import nibabel as nib

def gather_files(folder_pred, folder_ref, file_ending=".nii.gz", chill=False):
    files_pred = subfiles(folder_pred.replace('_IR',''), suffix=file_ending, join=False)# JJ
    files_prob = subfiles(folder_pred, suffix=".npz", join=False)  # for probs
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)

    if not chill:
        present = [isfile(join(folder_pred.replace('_IR',''), i)) for i in files_ref]# JJ
        assert all(present), "Not all files in folder_ref exist in folder_pred"

    files_ref = [join(folder_ref, i) for i in files_pred]
    files_prob = [join(folder_pred, i.replace("nii.gz", "npz")) for i in files_pred]
    files_pred = [join(folder_pred.replace('_IR', ''), i) for i in files_pred]  # JJ

    return files_pred, files_prob, files_ref


def gather_files_seed(folder_pred, folder_ref, file_ending=".nii.gz"):
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)  # JJ
    files_prob = subfiles(folder_pred, suffix=".npz", join=False)  # for probs
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_prob = [join(folder_pred, i.replace("nii.gz", "npz")) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    return files_pred, files_prob, files_ref

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:  # 1,2,3
            mask[segmentation == r] = True
    return mask, np.count_nonzero(mask)


def region_or_label_to_mask_prob_add(segmentation: np.ndarray,
                                     region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros(segmentation.shape[1:])
        for r in region_or_label:  # 1,2,3
            mask = mask + segmentation[r, :, :, :]  # mask[z,x,y], seg:4-channel
    return mask, np.sum(mask)


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def jaccard_segment(x1, y1, x2, y2):
    """directed line seg (x1, y1) (x2, y2) """
    intersection = max(0, min(y1, y2) - max(x1, x2))  # 交集长度
    union = max(y1, y2) - min(x1, x2)  # 并集长度
    return intersection / union if union > 0 else 0


def get_ce_bound(y_bar, x_bar, epsilon_y, epsilon_x):
    ce_left = y_bar / x_bar - max(y_bar - epsilon_y, 1e-7) / (x_bar + epsilon_x)  # extreme-case: move to 0
    ce_right = (y_bar + epsilon_y) / max(x_bar - epsilon_x, 1e-7) - y_bar / x_bar  # move to 1

    # if ce_right + ce_left < 0:
    #     import pdb;pdb.set_trace()
    return ce_left, ce_right


def convert_labels_to_one_hot(labels, num_classes=4):
    return F.one_hot(labels.long(), num_classes=num_classes).bool()  # 返回布尔类型

def cat_all_files(preds_source_list, preds_target_list, labels_source_list):
    print(f'concat for {len(preds_source_list)} preds_source ...')
    preds_list = []
    for file in preds_source_list:
        data = np.load(file)['probabilities']  # (4, 155, 240, 240)
        data = torch.from_numpy(data).float()  # to Tensor
        data = data.permute(1, 2, 3, 0).reshape(-1, 4)  # → (155, 240, 240, 4)
        preds_list.append(data)
    preds_source = torch.cat(preds_list, dim=0)

    print(f'concat for {len(preds_target_list)} preds_target ...')
    preds_list = []
    for file in preds_target_list:
        data = np.load(file)['probabilities']  # (4, 155, 240, 240)
        data = torch.from_numpy(data).float()  # to Tensor
        data = data.permute(1, 2, 3, 0).reshape(-1, 4)  # → (155, 240, 240, 4)
        preds_list.append(data)
    preds_target = torch.cat(preds_list, dim=0)

    print(f'concat for {len(labels_source_list)} labels_source ...')
    preds_list = []
    for file in labels_source_list:
        data = nib.load(file).get_fdata()
        data = torch.from_numpy(data).float()
        data = data.reshape(-1)
        preds_list.append(data)
    labels_source = torch.cat(preds_list, dim=0)  # cat for (N * 155*240*240)

    return preds_source, preds_target, labels_source

def cat_all_source_files(preds_source_list, labels_source_list):
    print(f'concat for {len(preds_source_list)} preds_source ...')
    preds_list = []
    for file in preds_source_list:
        data = np.load(file)['probabilities']  # (4, 155, 240, 240)
        data = torch.from_numpy(data).float()  # to Tensor
        data = data.permute(1, 2, 3, 0).reshape(-1, 4)  # → (155, 240, 240, 4)
        preds_list.append(data)
    preds_source = torch.cat(preds_list, dim=0)

    print(f'concat for {len(labels_source_list)} labels_source ...')
    preds_list = []
    for file in labels_source_list:
        data = nib.load(file).get_fdata()
        data = torch.from_numpy(data).float()
        data = data.reshape(-1)
        preds_list.append(data)
    labels_source = torch.cat(preds_list, dim=0)  # cat for (N * 155*240*240)

    return preds_source, labels_source

import multiprocessing
import os
from pathlib import Path
import re
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import torch
from nnunetv2.evaluation.ece_utils import fast_ece, brier_score, lp_score, l1_score
from nnunetv2.evaluation.ece_kde import get_ece_kde
from nnunetv2.evaluation.plot_utils import plot_r_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F

def gather_files(folder_pred, folder_ref, file_ending=".nii.gz", chill=False):
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_prob = subfiles(folder_pred, suffix=".npz", join=False)  # for probs
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)

    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"

    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    files_prob = [join(folder_pred, i.replace("nii.gz", "npz")) for i in files_pred]

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
    ce_left = y_bar / x_bar - max(y_bar - epsilon_y, 0) / (x_bar + epsilon_x)  # extreme-case: move to 0
    ce_right = (y_bar + epsilon_y) / max(x_bar - epsilon_x, 0) - y_bar / x_bar  # move to 1
    return ce_left, ce_right


def analyze_r_ce_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                     tensor_seg_gt_map, ce_type):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,n)
    var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    sigma_r = var_r ** 0.5

    #####################
    if 'kde' in ce_type:
        p = int(re.search(r'\d+', ce_type).group())
        epsilon_y, epsilon_x = calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,tensor_wt_gt_map, p)
    elif 'bins' in ce_type:
        bins = int(re.search(r'\d+', ce_type).group())
        epsilon_y, epsilon_x = calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,tensor_wt_gt_map, bins)
    elif ce_type == 'bs':
        epsilon_y, epsilon_x = calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,tensor_wt_gt_map)
        return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), 0, 0, sigma_r.item(), epsilon_y.item(), epsilon_x.item()
    elif ce_type == 'nll':
        epsilon_y, epsilon_x = calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    #####################
    ce_left, ce_right = get_ce_bound(y_bar, x_bar, epsilon_y, epsilon_x)

    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), ce_left.item(), ce_right.item(), sigma_r.item(), epsilon_y.item(), epsilon_x.item()


def analyze_r_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                  tensor_seg_gt_map, epsilon_y, epsilon_x):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,
                                                     n)
    var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    sigma_r = var_r ** 0.5

    ce_left = y_bar / x_bar - max(y_bar - epsilon_y, 0) / (x_bar + epsilon_x)  # extreme-case: move to 0
    ce_right = (y_bar + epsilon_y) / max(x_bar - epsilon_x, 0) - y_bar / x_bar  # move to 1

    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), ce_left.item(), ce_right.item(), sigma_r.item()


def calc_statistic(tensor_nec_prob_map, tensor_wt_prob_map):
    # mean/var
    nec_flat, wt_flat = tensor_nec_prob_map.reshape(-1), tensor_wt_prob_map.reshape(-1)
    n = nec_flat.numel()
    y_bar, x_bar = torch.mean(nec_flat), torch.mean(wt_flat)
    var_y, var_x = torch.var(nec_flat), torch.var(wt_flat)  # Using population variance to match NumPy
    cov_x_y = torch.cov(torch.stack((wt_flat, nec_flat)))[0, 1]
    x2_bar = torch.mean(wt_flat ** 2)
    y2_bar = torch.mean(nec_flat ** 2)
    cov_x_y = torch.mean((wt_flat - x_bar) * (nec_flat - y_bar))
    cov_x2_y = torch.mean((wt_flat ** 2 - x2_bar) * (nec_flat - y_bar))
    cov_y2_x = torch.mean((nec_flat ** 2 - y2_bar) * (wt_flat - x_bar))
    cov_x2_x = torch.mean((wt_flat ** 2 - x2_bar) * (wt_flat - x_bar))
    return y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n


def estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n):
    # print("Analyzing r_naive ...")
    r_naive = y_bar / x_bar
    # print("Analyzing r_2_ord_corr ...")
    ## r_a*, r_b*
    r_a = cov_x_y / (x_bar * y_bar)
    r_a_star = r_a * (1 + (1 / (n - 1)) * ((y_bar * cov_x2_y + x_bar * cov_y2_x) / (cov_x_y * x_bar * y_bar) - 4) - (
            1 / (n - 1)) * (var_x / x_bar ** 2 + var_y / y_bar ** 2 + 2 * cov_x_y / (x_bar * y_bar)))
    r_b = var_x / x_bar ** 2
    r_b_star = r_b * (
            1 + (4 / (n - 1)) * ((0.5 * cov_x2_x) / (x_bar * var_x) - 1) - (4 / (n - 1)) * (var_x / x_bar ** 2))
    ## r_corr
    r_1_ord_corr = r_naive * (1 - (1 / n) * (r_b_star - r_a_star))
    r_2_ord_corr = r_naive * (1 - (1 / n) * (r_b_star - r_a_star) - (1 / n ** 2) * (
            (cov_x2_y - 2 * x_bar * cov_x_y) / (x_bar ** 2 * y_bar) - (cov_x2_x - 2 * x_bar * var_x) / (x_bar ** 3) - (
            3 * var_x * cov_x_y) / (x_bar ** 3 * y_bar) + (3 * var_x ** 2) / (x_bar ** 4)))
    return r_naive, r_1_ord_corr, r_2_ord_corr


def get_ece_kde_sub(f, y, bandwidth, p, mc_type, device, sub=1e4):
    # random permute -> batch -> average
    idx = torch.randperm(f.shape[0])  # 例如 tensor([3, 1, 7, ..., 0])
    f_shuffled, y_shuffled = torch.clamp(f[idx, :], min=0, max=1), y[idx]
    batch_size = int(sub)  # enough
    batch_f = f_shuffled[:batch_size].to(device)
    batch_y = y_shuffled[:batch_size].to(device)
    batch_ratio = get_ece_kde(batch_f, batch_y, bandwidth, p, mc_type, device).to("cpu")
    return batch_ratio


def get_ece_bins(f, y, bins, device):
    f, y = f.squeeze(1).to(device), y.to(device)
    return fast_ece(f, y, n_bins=bins, device=device).to("cpu")
    # batch_ratio = fast_ece(f, y, n_bins=10, device=device).to("cpu").item()
    # return torch.tensor(batch_ratio)


def calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, p):
    print(f"Analyzing ece_kde with l-{p}...")
    # 1d_kde
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    bandwidth = 0.02  # 0.001
    device = "cuda"
    sub = 1e4
    epsilon_y = get_ece_kde_sub(tensor_nec_prob_map, tensor_nec_gt_map, bandwidth, p,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs 2
    epsilon_x = get_ece_kde_sub(tensor_wt_prob_map.to(device), tensor_wt_gt_map.to(device), bandwidth, p,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs {1,2,3}
    return epsilon_y, epsilon_x


def calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing Brier_score ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1), tensor_wt_prob_map.reshape(-1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(-1).to(torch.int64)
    epsilon_y = brier_score(tensor_nec_prob_map, tensor_nec_gt_map)
    epsilon_x = brier_score(tensor_wt_prob_map, tensor_wt_gt_map)
    return epsilon_y, epsilon_x

def calc_v_bias(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    v_bias_y = l1_score(tensor_nec_prob_map.reshape(-1), tensor_nec_gt_map.reshape(-1).to(torch.int64))
    v_bias_x = l1_score(tensor_wt_prob_map.reshape(-1), tensor_wt_gt_map.reshape(-1).to(torch.int64))
    return v_bias_y.item(), v_bias_x.item()

def calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing NLL ...")
    tensor_wt_prob_map = torch.clamp(tensor_wt_prob_map, min=0, max=1)
    epsilon_y = F.binary_cross_entropy(tensor_nec_prob_map.reshape(-1), tensor_nec_gt_map.reshape(-1).double(),
                                       reduction='mean')
    epsilon_x = F.binary_cross_entropy(tensor_wt_prob_map.reshape(-1), tensor_wt_gt_map.reshape(-1).double(),
                                       reduction='mean')
    return epsilon_y, epsilon_x


def calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, bins):
    print(f"Analyzing ece_bins ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    epsilon_y = get_ece_bins(tensor_nec_prob_map, tensor_nec_gt_map, bins=bins, device="cuda")  # binary: 0 vs 2
    epsilon_x = get_ece_bins(tensor_wt_prob_map, tensor_wt_gt_map, bins=bins, device="cuda")   # binary: 0 vs {1,2,3}
    return epsilon_y, epsilon_x


def clip_to_unit_range(x):
    return np.clip(x, 0, 1)  # range [0, 1]



def detect_failure(paired_samples):  # fail_case: outside the range
    fail_case = []
    r_gt = paired_samples['r_gt']['r_gt']
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+1std'][:, 0], paired_samples['r_naive']['bound__ce+1std'][:, 1]
    mask_1 = (r_gt < lower_std) | (r_gt > upper_std)
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+2std'][:, 0], paired_samples['r_naive']['bound__ce+2std'][:, 1]
    mask_2 = (r_gt < lower_std) | (r_gt > upper_std)
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+3std'][:, 0], paired_samples['r_naive']['bound__ce+3std'][:, 1]
    mask_3 = (r_gt < lower_std) | (r_gt > upper_std)
    case_ids = np.array(
        [os.path.basename(name).split('_')[-1].split('.')[0] for name in paired_samples['reference_file']])
    return case_ids[mask_1], case_ids[mask_2], case_ids[mask_3]

def update_r_keys(results, key, r_naive, r_gt, x0_ce, y0_ce, x0, y0, x0_2sigma, y0_2sigma, x0_3sigma, y0_3sigma):
    results['ratio'][key]['r_est'] = r_naive
    results['ratio'][key]['r_bias'] = r_naive - r_gt
    results['ratio'][key]['bound__ce'] = np.array([x0_ce, y0_ce])
    results['ratio'][key]['bound__ce+1std'] = np.array([x0, y0])
    results['ratio'][key]['bound__ce+2std'] = np.array([x0_2sigma, y0_2sigma])
    results['ratio'][key]['bound__ce+3std'] = np.array([x0_3sigma, y0_3sigma])
    results['ratio'][key]['range__ce+1std'] = y0 - x0
    results['ratio'][key]['range__ce+2std'] = y0_2sigma - x0_2sigma
    results['ratio'][key]['range__ce+3std'] = y0_3sigma - x0_3sigma
    return results


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      biomarker: str ='ntr',
                      ce_type: str = "bins") -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    print(f"calculate r for {os.path.basename(reference_file)}")
    interested_region = (2,) if biomarker=='ntr' else (2, 3)

    "analyze mean/var"
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, interested_region) # JJ, ntr, ctr

    # pred
    if binary:
        print("Hey, now we binarize preds for seg (not prob)")
        wt_prob_map, _ = region_or_label_to_mask(seg_pred, (1, 2, 3))
        nec_prob_map, _ = region_or_label_to_mask(seg_pred, interested_region)
        wt_prob_map = wt_prob_map.squeeze(0).astype(float)
        nec_prob_map = nec_prob_map.squeeze(0).astype(float)

    else:
        wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
        nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, interested_region)  # numerator

    # prob+gt into tensor
    tensor_nec_prob_map = torch.from_numpy(nec_prob_map)  # [155,240,240]
    tensor_wt_prob_map = torch.from_numpy(wt_prob_map)
    tensor_nec_gt_map = torch.from_numpy(nec_gt_map).squeeze(0)
    tensor_wt_gt_map = torch.from_numpy(wt_gt_map).squeeze(0)

    tensor_seg_prob_map = torch.from_numpy(prob_pred)
    tensor_seg_gt_map = torch.from_numpy(seg_ref)

    "reformulate r to see how interval(x,y) changes"
    ## V-Bias
    v_bias_y_L1, v_bias_x_L1 = calc_v_bias(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    # sigma
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    # calib-error（epsilon_y, epsilon_x）
    r_naive, r_1_ord_corr, r_2_ord_corr, ce_left, ce_right, sigma_r, epsilon_y, epsilon_x = analyze_r_ce_std(
        tensor_nec_prob_map,
        tensor_wt_prob_map,
        tensor_nec_gt_map,
        tensor_wt_gt_map,
        tensor_seg_prob_map,
        tensor_seg_gt_map,
        ce_type)
    # print(f'ce_l: {ce_left}, ce_r: {ce_right}') # 0.109, 0.162
    # print(f'std: {sigma_r}') # 0.001
    # CE_range
    # a prior is: r>0, which is appicable for confidence range
    x0_ce, y0_ce = clip_to_unit_range(r_naive - ce_left), clip_to_unit_range(r_naive + ce_right)
    x1_ce, y1_ce = clip_to_unit_range(r_1_ord_corr - ce_left), clip_to_unit_range(r_1_ord_corr + ce_right)
    x2_ce, y2_ce = clip_to_unit_range(r_2_ord_corr - ce_left), clip_to_unit_range(r_2_ord_corr + ce_right)
    # CE+sigma_range
    x0, y0 = clip_to_unit_range(x0_ce - sigma_r), clip_to_unit_range(y0_ce + sigma_r)  # naive_r, 1std
    x1, y1 = clip_to_unit_range(x1_ce - sigma_r), clip_to_unit_range(y1_ce + sigma_r)  # 1_order_corr_r
    x2, y2 = clip_to_unit_range(x2_ce - sigma_r), clip_to_unit_range(y2_ce + sigma_r)  # 2_order_corr_r
    x0_2sigma, y0_2sigma = clip_to_unit_range(x0 - sigma_r), clip_to_unit_range(y0 + sigma_r)
    x1_2sigma, y1_2sigma = clip_to_unit_range(x1 - sigma_r), clip_to_unit_range(y1 + sigma_r)
    x2_2sigma, y2_2sigma = clip_to_unit_range(x2 - sigma_r), clip_to_unit_range(y2 + sigma_r)
    x0_3sigma, y0_3sigma = clip_to_unit_range(x0 - 2 * sigma_r), clip_to_unit_range(y0 + 2 * sigma_r)  # naive_r, 3std
    x1_3sigma, y1_3sigma = clip_to_unit_range(x1 - 2 * sigma_r), clip_to_unit_range(y1 + 2 * sigma_r)
    x2_3sigma, y2_3sigma = clip_to_unit_range(x2 - 2 * sigma_r), clip_to_unit_range(y2 + 2 * sigma_r)
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'v_bias': {}, f'epsilon_ece_{ce_type}': {},
                        'r_naive': {}, 'r_first_corr': {},'r_second_corr': {},
                        'iou_scores': {}}
    # gt
    r_gt = nec_gt_counter / wt_gt_counter
    results['ratio']['r_gt']['r_gt'] = r_gt
    results['ratio']['r_gt']['sigma_r'] = sigma_r
    # V-Bias: y,x
    # v_bias
    results['ratio']['v_bias']['v_bias_y_L1'] = v_bias_y_L1
    results['ratio']['v_bias']['v_bias_x_L1'] = v_bias_x_L1
    # results['ratio']['v_bias']['v_bias_y_L2'] = v_bias_y_L2
    # results['ratio']['v_bias']['v_bias_x_L2'] = v_bias_x_L2
    # CE: y,x
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] = epsilon_y
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] = epsilon_x
    # estimation
    update_r_keys(results, 'r_naive', r_naive, r_gt, x0_ce, y0_ce, x0, y0, x0_2sigma, y0_2sigma, x0_3sigma, y0_3sigma)
    update_r_keys(results, 'r_first_corr', r_1_ord_corr, r_gt, x1_ce, y1_ce, x1, y1, x1_2sigma, y1_2sigma, x1_3sigma, y1_3sigma)
    update_r_keys(results, 'r_second_corr', r_2_ord_corr, r_gt, x1_ce, y2_ce, x2, y2, x2_2sigma, y2_2sigma, x2_3sigma,y2_3sigma)

    # Jaccard(r_naive,r_corr)
    naive_second_1 = jaccard_segment(x0, y0, x2, y2)
    naive_second_2 = jaccard_segment(x0_2sigma, y0_2sigma, x2_2sigma, y2_2sigma)
    naive_second_3 = jaccard_segment(x0_3sigma, y0_3sigma, x2_3sigma, y2_3sigma)
    results['ratio']['iou_scores']['naive_vs_second__ce+123std'] = np.array(
        [naive_second_1, naive_second_2, naive_second_3])
    return results


def compute_estimator_avg(reference_file: str, prediction_file: str, probability_file: str,
                          image_reader_writer: BaseReaderWriter,
                          labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                          ignore_label: int = None,
                          binary: bool = False,
                          biomarker: str ='ntr',
                          ce_type: str = "kde",
                          epsilon_y=None, epsilon_x=None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None


    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'v_bias': {}, f'epsilon_ece_{ce_type}': {},
                        'r_naive': {}, 'r_first_corr': {},'r_second_corr': {},
                        'iou_scores': {}}
    print(f"calculate r for {os.path.basename(reference_file)}")
    interested_region = (2,) if biomarker=='ntr' else (2, 3)

    "analyze mean/var"
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, interested_region) # JJ, ntr, ctr
    ## binary
    if binary:
        print("Hey, now we binarize preds for seg (not prob)")
        wt_prob_map, _ = region_or_label_to_mask(seg_pred, (1, 2, 3))
        nec_prob_map, _ = region_or_label_to_mask(seg_pred, interested_region)
        wt_prob_map = wt_prob_map.squeeze(0).astype(float)
        nec_prob_map = nec_prob_map.squeeze(0).astype(float)

    else:
        wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
        nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, interested_region)  # numerator

    # prob+gt into tensor
    tensor_nec_prob_map = torch.from_numpy(nec_prob_map)  # [155,240,240]
    tensor_wt_prob_map = torch.from_numpy(wt_prob_map)
    tensor_nec_gt_map = torch.from_numpy(nec_gt_map).squeeze(0)
    tensor_wt_gt_map = torch.from_numpy(wt_gt_map).squeeze(0)
    tensor_seg_prob_map = torch.from_numpy(prob_pred)
    tensor_seg_gt_map = torch.from_numpy(seg_ref)


    "reformulate r to see how interval(x,y) changes"
    ## V-Bias
    v_bias_y_L1, v_bias_x_L1 = calc_v_bias(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    # sigma
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    # calib-error（epsilon_y, epsilon_x）
    r_naive, r_1_ord_corr, r_2_ord_corr, ce_left, ce_right, sigma_r = analyze_r_std(
        tensor_nec_prob_map,
        tensor_wt_prob_map,
        tensor_nec_gt_map,
        tensor_wt_gt_map,
        tensor_seg_prob_map,
        tensor_seg_gt_map,
        epsilon_y, epsilon_x)

    # CE_range
    # a prior is: r>0, which is appicable for confidence range
    x0_ce, y0_ce = clip_to_unit_range(r_naive - ce_left), clip_to_unit_range(r_naive + ce_right)
    x1_ce, y1_ce = clip_to_unit_range(r_1_ord_corr - ce_left), clip_to_unit_range(r_1_ord_corr + ce_right)
    x2_ce, y2_ce = clip_to_unit_range(r_2_ord_corr - ce_left), clip_to_unit_range(r_2_ord_corr + ce_right)
    # CE+sigma_range
    x0, y0 = clip_to_unit_range(x0_ce - sigma_r), clip_to_unit_range(y0_ce + sigma_r)  # naive_r, 1std
    x1, y1 = clip_to_unit_range(x1_ce - sigma_r, ), clip_to_unit_range(y1_ce + sigma_r)  # 1_order_corr_r
    x2, y2 = clip_to_unit_range(x2_ce - sigma_r, ), clip_to_unit_range(y2_ce + sigma_r)  # 2_order_corr_r
    x0_2sigma, y0_2sigma = clip_to_unit_range(x0 - sigma_r), clip_to_unit_range(y0 + sigma_r)
    x1_2sigma, y1_2sigma = clip_to_unit_range(x1 - sigma_r), clip_to_unit_range(y1 + sigma_r)
    x2_2sigma, y2_2sigma = clip_to_unit_range(x2 - sigma_r), clip_to_unit_range(y2 + sigma_r)
    x0_3sigma, y0_3sigma = clip_to_unit_range(x0 - 2 * sigma_r), clip_to_unit_range(y0 + 2 * sigma_r)  # naive_r, 3std
    x1_3sigma, y1_3sigma = clip_to_unit_range(x1 - 2 * sigma_r), clip_to_unit_range(y1 + 2 * sigma_r)
    x2_3sigma, y2_3sigma = clip_to_unit_range(x2 - 2 * sigma_r), clip_to_unit_range(y2 + 2 * sigma_r)
    #####################################
    # gt
    r_gt = nec_gt_counter / wt_gt_counter
    results['ratio']['r_gt']['r_gt'] = r_gt
    results['ratio']['r_gt']['sigma_r'] = sigma_r
    # v_bias
    results['ratio']['v_bias']['v_bias_y_L1'] = v_bias_y_L1
    results['ratio']['v_bias']['v_bias_x_L1'] = v_bias_x_L1
    # results['ratio']['v_bias']['v_bias_y_L2'] = v_bias_y_L2
    # results['ratio']['v_bias']['v_bias_x_L2'] = v_bias_x_L2
    # CE: y,x
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] = epsilon_y
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] = epsilon_x
    # estimation
    update_r_keys(results, 'r_naive', r_naive, r_gt, x0_ce, y0_ce, x0, y0, x0_2sigma, y0_2sigma, x0_3sigma, y0_3sigma)
    update_r_keys(results, 'r_first_corr', r_1_ord_corr, r_gt, x1_ce, y1_ce, x1, y1, x1_2sigma, y1_2sigma, x1_3sigma, y1_3sigma)
    update_r_keys(results, 'r_second_corr', r_2_ord_corr, r_gt, x1_ce, y2_ce, x2, y2, x2_2sigma, y2_2sigma, x2_3sigma,y2_3sigma)

    naive_second_1 = jaccard_segment(x0, y0, x2, y2)
    naive_second_2 = jaccard_segment(x0_2sigma, y0_2sigma, x2_2sigma, y2_2sigma)
    naive_second_3 = jaccard_segment(x0_3sigma, y0_3sigma, x2_3sigma, y2_3sigma)
    results['ratio']['iou_scores']['naive_vs_second__ce+123std'] = np.array(
        [naive_second_1, naive_second_2, naive_second_3])
    return results



def compute_estimator_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                                image_reader_writer: BaseReaderWriter,
                                file_ending: str,
                                regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                ignore_label: int = None,
                                num_processes: int = default_num_processes,
                                chill: bool = True,
                                binary: bool = False,
                                biomarker: str ='ntr',
                                ce_type: str = "bins",  # "kde", nll, bs
                                exp: int = None,  # diff exp for kde,
                                ) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    # i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        # i += 1
        # if i > 130 or i < 128: continue  # JJ: first two samples
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, biomarker, ce_type)
        results.append(result)
    ## Jaccard: overlap metrics
    paired_samples = {'r_gt':{},'r_naive':{}, 'r_first_corr':{}, 'r_second_corr':{},
                      f'epsilon_ece_{ce_type}':{},'v_bias':{}, 'naive_vs_second__ce+123std':{}}
    mean_r_jaccard = {}
    paired_samples['naive_vs_second__ce+123std']['iou_scores'] = np.array([item['ratio']['iou_scores']['naive_vs_second__ce+123std'] for item in results])
    mean_r_jaccard['naive_vs_second__ce+123std'] = np.mean(paired_samples['naive_vs_second__ce+123std']['iou_scores'], axis=0)

    ################################################
    # r_gt and ref.nii.gz
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    ## Range: as narrow as possible ##
    mean_r_range = {}
    mean_r_bias = {}
    scores = ['r_naive', 'r_first_corr', 'r_second_corr']
    for r in scores:
        paired_samples[r]['bound__ce'] = np.array([item['ratio'][r]['bound__ce'] for item in results])  # [N,2]
        paired_samples[r]['bound__ce+1std'] = np.array(
            [item['ratio'][r]['bound__ce+1std'] for item in results])  # [N,2]
        paired_samples[r]['bound__ce+2std'] = np.array([item['ratio'][r]['bound__ce+2std'] for item in results])
        paired_samples[r]['bound__ce+3std'] = np.array([item['ratio'][r]['bound__ce+3std'] for item in results])
        paired_samples[r]['range__ce+1std'] = np.array([item['ratio'][r]['range__ce+1std'] for item in results])  # [N,]
        paired_samples[r]['range__ce+2std'] = np.array([item['ratio'][r]['range__ce+2std'] for item in results])
        paired_samples[r]['range__ce+3std'] = np.array([item['ratio'][r]['range__ce+3std'] for item in results])
        paired_samples[r]['bias_r'] = np.array([item['ratio'][r]['r_bias'] for item in results])
        # paired_samples[r]['sigma_r'] = results['ratio']['r_gt']['sigma_r']
        paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
        r_range_123 = np.stack((paired_samples[r]['range__ce+1std'], paired_samples[r]['range__ce+2std'],
                                paired_samples[r]['range__ce+3std']), axis=1)
        mean_r_range[r] = np.mean(r_range_123, axis=0)
        mean_r_bias[r] = np.mean(paired_samples[r]['bias_r'])
    # cali-error for y and x
    mean_epsilon = {}
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] for item in results])
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] for item in results])
    mean_epsilon['epsilon_y'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'])
    # bias for y and x
    mean_v_bias = {}
    paired_samples['v_bias']['v_bias_y_L1'] = np.array([item['ratio']['v_bias']['v_bias_y_L1'] for item in results])
    paired_samples['v_bias']['v_bias_x_L1'] = np.array([item['ratio']['v_bias']['v_bias_x_L1'] for item in results])
    mean_v_bias['v_bias_y_L1'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'])
    mean_v_bias['v_bias_x_L1'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'])
    # failure
    failure_1std, failure_2std, failure_3std = detect_failure(paired_samples)  # fail_case: outside the range
    failure = {}
    failure['std'], failure['2std'], failure['3std'] = failure_1std.tolist(), failure_2std.tolist(), failure_3std.tolist()
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_jaccard, mean_r_range, mean_r_bias, mean_epsilon, mean_v_bias, failure]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_v_bias': mean_v_bias,
              'mean_r_bias': mean_r_bias,
              'mean_r_range__ce+123std': mean_r_range,
              'failure': failure,
              'mean_r_jaccard__ce+123std': mean_r_jaccard,
              'ratio_per_case': results,
              }

    if binary:
        folder_save = join(folder_pred, f"ratio_metrics_binary_{biomarker}")
    else:
        folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
        
    # ratio.json
    os.makedirs(folder_save, exist_ok=True)
    save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)
    
    # # plot.json
    result_plot = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
                   'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
    result_as_list = {
        key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
        key, value in result_plot.items()}  # extract bound-related items
    os.makedirs(join(folder_save,'plot'), exist_ok=True)
    save_json(result_as_list, join(folder_save,'plot', f"plot_{ce_type}_{output_file}"), sort_keys=False)
    ################################################
    if ce_type!='bs':
        sigmas = ['', 2, 3]
        step_size = 10
        for sigma in sigmas:
            plot_r_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # r, r_corr
            plot_ce_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # ce, ce+sigma
            plot_bins_dataset(paired_samples, folder_save, sigma, ce_type, exp)  # "hist_of bias_and_range": bias_r, range
    ################################################
    return paired_samples[f'epsilon_ece_{ce_type}']  # ['epsilon_ece_kde']['epsilon_y'] and 'x' for 2 np.arrays


def compute_estimator_on_folder_avg(folder_ref: str, folder_pred: str, output_file: str,
                                    image_reader_writer: BaseReaderWriter,
                                    file_ending: str,
                                    regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                    ignore_label: int = None,
                                    num_processes: int = default_num_processes,
                                    chill: bool = True,
                                    binary: bool = False,
                                    biomarker: str ='ntr',
                                    ce_type: str = "kde",  # "kde", nll, bs
                                    exp: str = "avg",  # diff exp for kde,
                                    avg_epsilon_y: Any = None,
                                    avg_epsilon_x: Any = None) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    for ref, pred, prob, epsilon_y, epsilon_x in zip(files_ref, files_pred, files_prob, avg_epsilon_y, avg_epsilon_x):
        # for ref, pred, prob, epsilon_y, epsilon_x in zip(files_ref[:3], files_pred[:3], files_prob[:3], avg_epsilon_y[:3], avg_epsilon_x[:]):
        result = compute_estimator_avg(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                       binary, biomarker, ce_type, epsilon_y, epsilon_x)
        results.append(result)
    ## Jaccard: overlap metrics
    paired_samples = {'r_gt':{},'r_naive':{}, 'r_first_corr':{}, 'r_second_corr':{},
                      f'epsilon_ece_{ce_type}':{},'v_bias':{}, 'naive_vs_second__ce+123std':{}}
    mean_r_jaccard = {}
    paired_samples['naive_vs_second__ce+123std'] = {}
    paired_samples['naive_vs_second__ce+123std']['iou_scores'] = np.array([item['ratio']['iou_scores']['naive_vs_second__ce+123std'] for item in results])
    mean_r_jaccard['naive_vs_second__ce+123std'] = np.mean(paired_samples['naive_vs_second__ce+123std']['iou_scores'], axis=0)

    ################################################
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    ## Range: as narrow as possible ##
    mean_r_range = {}
    mean_r_bias = {}
    scores = ['r_naive', 'r_first_corr', 'r_second_corr']
    for r in scores:
        paired_samples[r]['bound__ce'] = np.array([item['ratio'][r]['bound__ce'] for item in results])  # [N,2]
        paired_samples[r]['bound__ce+1std'] = np.array(
            [item['ratio'][r]['bound__ce+1std'] for item in results])  # [N,2]
        paired_samples[r]['bound__ce+2std'] = np.array([item['ratio'][r]['bound__ce+2std'] for item in results])
        paired_samples[r]['bound__ce+3std'] = np.array([item['ratio'][r]['bound__ce+3std'] for item in results])
        paired_samples[r]['range__ce+1std'] = np.array([item['ratio'][r]['range__ce+1std'] for item in results])  # [N,]
        paired_samples[r]['range__ce+2std'] = np.array([item['ratio'][r]['range__ce+2std'] for item in results])
        paired_samples[r]['range__ce+3std'] = np.array([item['ratio'][r]['range__ce+3std'] for item in results])
        # paired_samples[r]['sigma_r'] = results['ratio']['r_gt']['sigma_r']
        paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
        r_range_123 = np.stack((paired_samples[r]['range__ce+1std'], paired_samples[r]['range__ce+2std'],
                                paired_samples[r]['range__ce+3std']), axis=1)
        mean_r_range[r] = np.mean(r_range_123, axis=0)
        # paired_samples[r]['bias_r'] = paired_samples[r]['r_est'] - paired_samples['r_gt']['r_gt']
        paired_samples[r]['bias_r'] = np.array([item['ratio'][r]['r_bias'] for item in results])
        mean_r_bias[r] = np.mean(paired_samples[r]['bias_r'])
    # cali-error for y and x
    mean_epsilon = {}
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] for item in results])
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] for item in results])
    mean_epsilon['epsilon_y'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'])
    failure_1std, failure_2std, failure_3std = detect_failure(paired_samples)  # fail_case: outside the range
    failure = {}
    failure['std'], failure['2std'], failure['3std'] = failure_1std, failure_2std, failure_3std
    # bias for y and x
    mean_v_bias = {}
    paired_samples['v_bias']['v_bias_y_L1'] = np.array([item['ratio']['v_bias']['v_bias_y_L1'] for item in results])
    paired_samples['v_bias']['v_bias_x_L1'] = np.array([item['ratio']['v_bias']['v_bias_x_L1'] for item in results])
    mean_v_bias['v_bias_y_L1'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'])
    mean_v_bias['v_bias_x_L1'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'])
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_jaccard, mean_r_range, mean_r_bias, mean_epsilon, mean_v_bias, failure]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_v_bias': mean_v_bias,
              'mean_r_bias': mean_r_bias,
              'mean_r_range__ce+123std': mean_r_range,
              'failure': failure,
              'mean_r_jaccard__ce+123std': mean_r_jaccard,
              'ratio_per_case': results,
              }

    if binary:
        folder_save = join(folder_pred, f"ratio_metrics_binary_{biomarker}")
    else:
        folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
    # ratio.json
    os.makedirs(folder_save, exist_ok=True)
    save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)
    
    # plot.json
    result_plot = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
                   'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
    result_as_list = {
        key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
        key, value in result_plot.items()}  # extract bound-related items
    os.makedirs(join(folder_save,'plot'), exist_ok=True)
    save_json(result_as_list, join(folder_save,'plot', f"plot_{ce_type}_{output_file}"), sort_keys=False)
    ################################################
    sigmas = ['', 2, 3]
    step_size = 10
    for sigma in sigmas:
        plot_r_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # r, r_corr
        plot_ce_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # ce, ce+sigma
        plot_bins_dataset(paired_samples, folder_save, sigma, ce_type, exp)  # "hist_of bias_and_range": bias_r, range

    ################################################
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               binary: bool = False,
                               biomarker: str ='ntr',
                               ce_type: str = "bins"  # "kde", nll, bs
                               ):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']  # .nii.gz for segs
    # file_ending = ".npz" #.npz for probs

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = 'ratio.json'

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    repeat_for_kde = 5

    if 'kde' in ce_type:
        ## repeate 5 times
        list_epsilon_y, list_epsilon_x = [], []
        for exp in range(repeat_for_kde):
            print(f'JJ: repeate on time {exp}...')
            output_file_exp = output_file.replace('.json', f"_1e4_{exp}.json")  # 1e4
            ##  ['epsilon_ece_kde']['epsilon_y'] and 'x'  for 2 np.arrays
            epsilon_dict = compute_estimator_on_folder(folder_ref, folder_pred, output_file_exp, rw, file_ending,
                                                       lm.foreground_regions if lm.has_regions else lm.foreground_labels,
                                                       lm.ignore_label,
                                                       num_processes, chill=chill, binary=binary,biomarker=biomarker,
                                                       ce_type=ce_type,exp=exp)
            list_epsilon_y.append(epsilon_dict['epsilon_y'])
            list_epsilon_x.append(epsilon_dict['epsilon_x'])
        ## avg
        avg_epsilon_y, avg_epsilon_x = np.mean(list_epsilon_y, axis=0), np.mean(list_epsilon_x, axis=0)
        exp = 'avg'
        print(f'JJ: Final {exp}...')
        output_file_exp = output_file.replace('.json', f"_1e4_{exp}.json")  # exp=avg
        compute_estimator_on_folder_avg(folder_ref, folder_pred, output_file_exp, rw, file_ending,
                                        lm.foreground_regions if lm.has_regions else lm.foreground_labels,
                                        lm.ignore_label,
                                        num_processes, chill=chill, binary=binary,
                                        ce_type=ce_type, exp=exp,
                                        avg_epsilon_y=avg_epsilon_y, avg_epsilon_x=avg_epsilon_x)

    else:
        compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                    lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                    num_processes, chill=chill, binary=binary, biomarker=biomarker, ce_type=ce_type, exp=None)




def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=False,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=False,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true',
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')
    parser.add_argument('--binary', action='store_true', help='use seg instead of prob for ratio est')
    parser.add_argument('--TS', type=str, required=False, default=None, help='Temperature Scaling')
    parser.add_argument('--ce_type', required=True, type=str, help='bins15, kde1, nll, bs')
    parser.add_argument('--biomarker', required=True, type=str, help='ntr, ctr')
    args = parser.parse_args()
    if args.pfile is None:
        args.pfile = Path(args.pred_folder.rstrip("/")).parents[1] / "plans.json"
    if args.djfile is None:
        args.djfile = Path(args.pred_folder.rstrip("/")).parents[1] / "dataset.json"
    if args.TS is not None:
        basename = os.path.basename(args.pred_folder)
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_TS_{args.TS}')
        print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, binary=args.binary, biomarker=args.biomarker,ce_type=args.ce_type)




if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions,
                              ignore_label,
                              num_processes)

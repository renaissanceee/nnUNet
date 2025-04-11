import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional
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
from nnunetv2.evaluation.fast_ece import fast_ece
from nnunetv2.evaluation.ece_kde import get_ece_kde
from nnunetv2.evaluation.plot_utils import plot_histogram_bias, plot_histogram_range, plot_r_and_range, plot_ce_and_range

def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    results_converted['mean_r_bias'] = {label_or_region_to_key(k): results['mean_r_bias'][k]
                                        for k in results['mean_r_bias'].keys()}
    results_converted['mean_r_jaccard__ce+123std'] = {label_or_region_to_key(k): results['mean_r_jaccard__ce+123std'][k]
                                                      for k in results['mean_r_jaccard__ce+123std'].keys()}
    results_converted['mean_r_range__ce+123std'] = {label_or_region_to_key(k): results['mean_r_range__ce+123std'][k] for
                                                    k in results['mean_r_range__ce+123std'].keys()}
    # convert ratio_per_case
    for i in range(len(results_converted["ratio_per_case"])):
        results_converted["ratio_per_case"][i]['ratio'] = \
            {label_or_region_to_key(k): results["ratio_per_case"][i]['ratio'][k]
             for k in results["ratio_per_case"][i]['ratio'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=False)  # sort_keys=True


def load_summary_json(filename: str):
    results = load_json(filename)
    results['mean_r_bias'] = {key_to_lgabel_or_region(k): results['mean_r_bias'][k] for k in
                              results['mean_r_bias'].keys()}
    results['mean_r_jaccard__ce+123std'] = {key_to_lgabel_or_region(k): results['mean_r_jaccard__ce+123std'][k] for k in
                                            results['mean_r_jaccard__ce+123std'].keys()}
    results['mean_r_range__ce+123std'] = {key_to_lgabel_or_region(k): results['mean_r_range__ce+123std'][k] for k in
                                          results['mean_r_range__ce+123std'].keys()}
    # convert ratio_per_case
    for i in range(len(results["ratio_per_case"])):
        results["ratio_per_case"][i]['ratio'] = \
            {key_to_label_or_region(k): results["ratio_per_case"][i]['ratio'][k]
             for k in results["ratio_per_case"][i]['ratio'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:  # 1,2,3
            mask[segmentation == r] = True
    return mask, np.count_nonzero(mask)


def region_or_label_to_mask_prob_max(segmentation: np.ndarray,
                                     region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(mask_ref)
        for r in region_or_label:  # 1,2,3
            cur_seg_channel_map = segmentation[r - 1, :, :, :][None, ...]
            mask = np.maximum(mask, cur_seg_channel_map)  # take max confidence over 3-channels
    mask[~mask_ref] = 0  # only consider tp pixels
    return mask, np.sum(mask)


def region_or_label_to_mask_prob_multiply(segmentation: np.ndarray,
                                          region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros(segmentation.shape[1:])
        for r in region_or_label:  # 1,2,3
            if r == region_or_label[0]:
                mask = segmentation[r - 1, :, :, :][None, ...]  # init as non-zero
            else:
                cur_seg_channel_map = segmentation[r - 1, :, :, :][None, ...]
                mask = mask * cur_seg_channel_map  # take prduction over 3-channels
    return mask, np.sum(mask)


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


def analyze_r_ce_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,
                                                     n)
    var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    sigma_r = var_r ** 0.5
    # 1d_kde
    epsilon_y, epsilon_x = calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    # debug
    ce_left = y_bar / x_bar - max(y_bar - epsilon_y, 0) / (x_bar + epsilon_x)  # extreme-case: move to 0
    ce_right = (y_bar + epsilon_y) / max(x_bar - epsilon_x, 0) - y_bar / x_bar  # move to 1
    # overall-range
    l_range = ce_left + sigma_r  # CE_left + σ
    r_range = ce_right + sigma_r  # CE_right + σ    # range__ce+1std = ce_left + ce_right + 2*sigma_r
    # return r_naive,r_1_ord_corr,r_2_ord_corr,ce_left,ce_right, sigma_r
    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), ce_left.item(), ce_right.item(), sigma_r.item(), epsilon_y.item(), epsilon_x.item()



def plot_r_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10):
    folder_save = join(folder_save, f"plots_interval_{step_size}_1e4", f"binaryCE_{sigma}sigma")
    os.makedirs(folder_save, exist_ok=True)

    r_est_naive = paired_samples['r_naive']['r_est']
    r_est_second = paired_samples['r_second_corr']['r_est']
    r_gt = paired_samples['r_gt']['r_gt']
    name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']
    bound_second = paired_samples['r_second_corr'][f'bound__ce+{sigma_c}std']
    for start in range(0, len(r_est_naive), step_size):
        end = start + step_size
        save_path = join(folder_save, f"r_and_range_{end}.png")
        plot_r_and_range(r_est_naive[start:end], r_est_second[start:end], r_gt[start:end], bound_naive[start:end],
                         bound_second[start:end], save_path, sigma, name_list[start:end])


def plot_ce_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10):
    folder_save = join(folder_save, f"plots_interval_{step_size}_1e4", f"binaryCE_{sigma}sigma_sep")
    os.makedirs(folder_save, exist_ok=True)

    r_est_naive = paired_samples['r_naive']['r_est']
    r_gt = paired_samples['r_gt']['r_gt']
    name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma 
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']
    bound_ce_naive = paired_samples['r_naive']['bound__ce']
    for start in range(0, len(r_est_naive), step_size):
        end = start + step_size
        save_path = join(folder_save, f"ce_and_range_{end}.png")
        plot_ce_and_range(r_est_naive[start:end], r_gt[start:end], bound_ce_naive[start:end], bound_naive[start:end],
                          save_path, sigma, name_list[start:end])

def plot_bins_dataset(paired_samples, folder_save, sigma=""):
    folder_save = join(folder_save, f"bins_interval_1e4")
    os.makedirs(folder_save, exist_ok=True)

    sigma_c = 1 if sigma=="" else sigma
    range_ce = paired_samples['r_naive'][f'range__ce+{sigma_c}std']
    bias = paired_samples['r_naive']['bias_r']
    save_path = join(folder_save, f"bias_hist_{sigma}sigma.png")
    plot_histogram_bias(bias, title=f'Histogram of Ratio Bias (±{sigma}$\\sigma$)',   # Bias
               xlabel='Ratio Bias', save_path=save_path, color='skyblue')
    plot_histogram_range(range_ce, title=f'Histogram of Confidence Interval (±{sigma}$\\sigma$)', # Interval
                   xlabel=f'Interval Length', save_path=save_path.replace('bias','range'), color='#A1D6B5')

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
    ## r_naive (y_bar/x_bar)
    print("Analyzing r_naive ...")
    r_naive = y_bar / x_bar
    ## (appendix C)
    print("Analyzing r_2_ord_corr ...")
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

def get_ece_kde_1e4_train(f, y, bandwidth, p, mc_type, device):
    # random permute -> batch -> average
    idx = torch.randperm(f.shape[0])  # 例如 tensor([3, 1, 7, ..., 0])
    f_shuffled, y_shuffled = torch.clamp(f[idx, :], min=0, max=1), y[idx]
    batch_size = int(1e4)  # enough
    batch_f = f_shuffled[:batch_size,:]
    batch_y = y_shuffled[:batch_size]
    batch_ratio = get_ece_kde(batch_f, batch_y, bandwidth, p, mc_type, device)
    return batch_ratio

def get_ece_kde_1e4(f, y, bandwidth, p, mc_type, device):
    # random permute -> batch -> average
    idx = torch.randperm(f.shape[0])  # 例如 tensor([3, 1, 7, ..., 0])
    f_shuffled, y_shuffled = torch.clamp(f[idx, :], min=0, max=1), y[idx]
    batch_size = int(1e4)  # enough
    batch_f = f_shuffled[:batch_size].to(device)
    batch_y = y_shuffled[:batch_size].to(device)
    batch_ratio = get_ece_kde(batch_f, batch_y, bandwidth, p, mc_type, device).to("cpu").item()
    return torch.tensor(batch_ratio)

def get_ece_kde_batch(f, y, bandwidth, p, mc_type, device):
    # random permute -> batch -> average
    idx = torch.randperm(f.shape[0])  # 例如 tensor([3, 1, 7, ..., 0])
    f_shuffled, y_shuffled = torch.clamp(f[idx, :], min=0, max=1), y[idx]
    batch_size = int(1e4)  # enough
    batch_ratio = []
    for i in range(0, len(f), batch_size):
        batch_f = f_shuffled[i:min(i + batch_size, len(f))].to(device)
        batch_y = y_shuffled[i:min(i + batch_size, len(y))].to(device)
        batch_ratio.append(get_ece_kde(batch_f, batch_y, bandwidth, p, mc_type, device).to("cpu").item())
    return torch.mean(torch.tensor(batch_ratio))


def calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing ece_kde ...")
    # 1d_kde
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    bandwidth = 0.02  # 0.001
    device = "cuda"
    # epsilon_y = get_ece_kde_batch(tensor_nec_prob_map.to(device), tensor_nec_gt_map.to(device), bandwidth, p=1,
    #                               mc_type='canonical', device=device)  # binary: 0 vs 2
    # epsilon_x = get_ece_kde_batch(tensor_wt_prob_map.to(device), tensor_wt_gt_map.to(device), bandwidth, p=1,
    #                               mc_type='canonical', device=device)  # binary: 0 vs {1,2,3}
    epsilon_y = get_ece_kde_1e4(tensor_nec_prob_map.to(device), tensor_nec_gt_map.to(device), bandwidth, p=1,
                                  mc_type='canonical', device=device)  # binary: 0 vs 2
    epsilon_x = get_ece_kde_1e4(tensor_wt_prob_map.to(device), tensor_wt_gt_map.to(device), bandwidth, p=1,
                                  mc_type='canonical', device=device)  # binary: 0 vs {1,2,3}
    # print(f"ece_kde_y,ece_kde_x: {epsilon_y},{epsilon_x}")
    return epsilon_y, epsilon_x


def downsample_3d_tensor(tensor, sampling_factor=2, random_seed=None):
    """
    对3D张量进行多维度降采样 (带随机种子控制)
    Args:
        tensor: 输入张量，形状为 [D, H, W] (此处为 [155, 240, 240])
        sampling_factor: 降采样因子 (默认2)
        random_seed: 随机种子 (None表示不固定)
    Returns:
        降采样后的张量，形状为 [D//f, H//f, W//f]
    """
    assert len(tensor.shape) == 3, "input must be [z,h,w]"
    D, H, W = tensor.shape
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        import random
        random.seed(random_seed)

    # z-slices: equal-interval
    dim0_indices = torch.linspace(0, D - 1, D // sampling_factor, dtype=torch.long)
    # H,W: 2-step-sampling
    dim1_indices = torch.randint(0, H, (H // sampling_factor,)).sort().values
    dim2_indices = torch.randint(0, W, (W // sampling_factor,)).sort().values
    sampled_tensor = tensor[dim0_indices[:, None, None],
    dim1_indices[None, :, None],
    dim2_indices[None, None, :]]

    return sampled_tensor


def clip_to_unit_range(x):
    """Clip the input value x to the range [0, 1] using NumPy."""
    return np.clip(x, 0, 1)


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'epsilon_ece_kde': {}, 'r_naive': {}, 'r_first_corr': {}, 'r_second_corr': {},
                        'iou_scores': {}}
    print(f"calculate r for {os.path.basename(reference_file)}")

    "analyze mean/var"
    labels_or_regions = [(2,), (1, 2, 3)]  # (2,3):core, (2,):necrosis
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, (2,))
    # print(np.sum(slice_data== 2),np.sum(slice_data== 1)+np.sum(slice_data== 2)+np.sum(slice_data== 3))
    # pred
    wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
    nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (2,))  # numerator

    # calib-error（epsilon_y, epsilon_x）
    # prob
    tensor_nec_prob_map = torch.from_numpy(nec_prob_map)  # [155,240,240]
    tensor_wt_prob_map = torch.from_numpy(wt_prob_map)
    # binary
    if binary:
        print("Hey, now we binarize preds for seg (not prob)")
        # seg: argmax for 1
        tensor_seg_pred = torch.from_numpy(seg_pred)
        tensor_nec_prob_map = (tensor_seg_pred==2).to(torch.float64)
        tensor_wt_prob_map = torch.isin(tensor_seg_pred, torch.tensor([1, 2, 3])).double()

    tensor_nec_gt_map = torch.from_numpy(nec_gt_map).squeeze(0)
    tensor_wt_gt_map = torch.from_numpy(wt_gt_map).squeeze(0)

    ################################
    # ## downsample + scattor
    # seeds = [10,20,30,40,50]
    # sampling_factor = 2
    # for seed in seeds:
    #     # downsample: no change distr
    #     downsampled_nec_prob_map = downsample_3d_tensor(tensor_nec_prob_map, sampling_factor, seed)# [155,240,240]->[77,120,120]
    #     downsampled_wt_prob_map = downsample_3d_tensor(tensor_wt_prob_map, sampling_factor, seed)
    #     # downsampled_nec_gt_map = downsample_3d_tensor(tensor_nec_gt_map, sampling_factor, seed)
    #     # downsampled_wt_gt_map = downsample_3d_tensor(tensor_wt_gt_map, sampling_factor, seed)
    #     # calc r
    #
    #     y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(downsampled_nec_prob_map,downsampled_wt_prob_map)
    #     down_r_naive, down_r_1_ord_corr, down_r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n)
    #     # Todo: save 5 scattor -> plot
    #
    ################################
    "reformulate r to see how interval(x,y) changes"
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr, ce_left, ce_right, sigma_r, epsilon_y, epsilon_x = analyze_r_ce_std(
        tensor_nec_prob_map,
        tensor_wt_prob_map,
        tensor_nec_gt_map,
        tensor_wt_gt_map)
    # a prior is: r>0, which is appicable for confidence range
    # CE_range
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
    # CE: y,x
    results['ratio']['epsilon_ece_kde']['epsilon_y'] = epsilon_y  # JJ
    results['ratio']['epsilon_ece_kde']['epsilon_x'] = epsilon_x
    ## naive
    results['ratio']['r_naive']['r_est'] = r_naive
    # [x,y]
    results['ratio']['r_naive']['bound__ce'] = np.array([x0_ce, y0_ce])
    results['ratio']['r_naive']['bound__ce+1std'] = np.array([x0, y0])
    results['ratio']['r_naive']['bound__ce+2std'] = np.array([x0_2sigma, y0_2sigma])
    results['ratio']['r_naive']['bound__ce+3std'] = np.array([x0_3sigma, y0_3sigma])
    # scalar
    results['ratio']['r_naive']['range__ce+1std'] = y0 - x0
    results['ratio']['r_naive']['range__ce+2std'] = y0_2sigma - x0_2sigma
    results['ratio']['r_naive']['range__ce+3std'] = y0_3sigma - x0_3sigma
    ## 1.order
    results['ratio']['r_first_corr']['r_est'] = r_1_ord_corr
    results['ratio']['r_first_corr']['bound__ce'] = np.array([x1_ce, y1_ce])
    results['ratio']['r_first_corr']['bound__ce+1std'] = np.array([x1, y1])
    results['ratio']['r_first_corr']['bound__ce+2std'] = np.array([x1_2sigma, y1_2sigma])
    results['ratio']['r_first_corr']['bound__ce+3std'] = np.array([x1_3sigma, y1_3sigma])
    results['ratio']['r_first_corr']['range__ce+1std'] = y1 - x1
    results['ratio']['r_first_corr']['range__ce+2std'] = y1_2sigma - x1_2sigma
    results['ratio']['r_first_corr']['range__ce+3std'] = y1_3sigma - x1_3sigma
    ## 2.order
    results['ratio']['r_second_corr']['r_est'] = r_2_ord_corr
    results['ratio']['r_second_corr']['bound__ce'] = np.array([x2_ce, y2_ce])
    results['ratio']['r_second_corr']['bound__ce+1std'] = np.array([x2, y2])
    results['ratio']['r_second_corr']['bound__ce+2std'] = np.array([x2_2sigma, y2_2sigma])
    results['ratio']['r_second_corr']['bound__ce+3std'] = np.array([x2_3sigma, y2_3sigma])
    results['ratio']['r_second_corr']['range__ce+1std'] = y2 - x2
    results['ratio']['r_second_corr']['range__ce+2std'] = y2_2sigma - x2_2sigma
    results['ratio']['r_second_corr']['range__ce+3std'] = y2_3sigma - x2_3sigma
    # Jaccard(r_naive,r_corr)
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
                                binary: bool = False) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_prob = subfiles(folder_pred, suffix=".npz", join=False)  # npz for probs
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    files_prob = [join(folder_pred, i.replace("nii.gz", "npz")) for i in files_pred]
    # with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
    #     # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
    #     #     compute_metrics(*i)
    #     results = pool.starmap(
    #         compute_metrics,
    #         list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
    #                  [ignore_label] * len(files_pred)))
    #     )

    results = []
    # i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        # i += 1
        # if i > 130 or i < 120: continue  # JJ: first two samples
        # if (ref!="/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/Dataset137_BraTS2021/labelsTs/fold_0/BraTS2021_01240.nii.gz"
        #         and ref!="/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/Dataset137_BraTS2021/labelsTs/fold_0/BraTS2021_00753.nii.gz"): continue
        # result = compute_metrics(ref, pred, image_reader_writer, regions_or_labels, ignore_label) # also do
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label, binary)
        results.append(result)

    ## Jaccard: overlap metrics
    paired_samples = {}
    mean_r_jaccard = {}
    scores = ['naive_vs_second__ce+123std', ]  # scores = ['naive_vs_first_123std','first_vs_second_123std']
    for r in scores:  # re-arrange
        paired_samples[r] = {}
        paired_samples[r]['iou_scores'] = np.array([item['ratio']['iou_scores'][r] for item in results])
        mean_r_jaccard[r] = np.mean(paired_samples[r]['iou_scores'], axis=0)

    ################################################
    # r_gt and ref.nii.gz
    paired_samples['r_gt'] = {}
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    ## Range: as narrow as possible ##
    mean_r_range = {}
    mean_r_bias = {}
    scores = ['r_naive', 'r_first_corr', 'r_second_corr']
    for r in scores:
        paired_samples[r] = {}
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
        paired_samples[r]['bias_r'] = paired_samples[r]['r_est'] - paired_samples['r_gt']['r_gt']
        mean_r_bias[r] = np.mean(paired_samples[r]['bias_r'])
    # cali-error for y and x
    mean_epsilon = {}
    paired_samples['epsilon_ece_kde'] = {}
    paired_samples['epsilon_ece_kde']['epsilon_y'] = np.array(
        [item['ratio']['epsilon_ece_kde']['epsilon_y'] for item in results])
    paired_samples['epsilon_ece_kde']['epsilon_x'] = np.array(
        [item['ratio']['epsilon_ece_kde']['epsilon_x'] for item in results])  # JJ
    mean_epsilon['epsilon_y'] = np.mean(paired_samples['epsilon_ece_kde']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples['epsilon_ece_kde']['epsilon_x'])

    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    means_results = [mean_r_jaccard, mean_r_range, mean_r_bias, mean_epsilon]
    [recursive_fix_for_json_export(i) for i in means_results]
    # recursive_fix_for_json_export(mean_r_jaccard)
    # recursive_fix_for_json_export(mean_r_range)
    # recursive_fix_for_json_export(mean_r_bias)
    # recursive_fix_for_json_export(mean_epsilon)
    result = {'mean_ece_kde': mean_epsilon, 'mean_r_bias': mean_r_bias, 'mean_r_jaccard__ce+123std': mean_r_jaccard,
              'mean_r_range__ce+123std': mean_r_range,
              'ratio_per_case': results}

    # write into: binaryCE_sigma_for_CrEDC_1e4.json
    folder_save = join(folder_pred, "ratio_metrics_binary") if binary else join(folder_pred, "ratio_metrics_prob")
    os.makedirs(folder_save, exist_ok=True)

    if output_file is not None:
        save_summary_json(result, join(folder_save, output_file))

    # fail_case: outside the range
    fail_case = []
    r_gt = paired_samples['r_gt']['r_gt']
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+1std'][:, 0], paired_samples['r_naive'][
                                                                                  'bound__ce+1std'][:, 1]
    mask = (r_gt < lower_std) | (r_gt > upper_std)
    case_ids = np.array(
        [os.path.basename(name).split('_')[-1].split('.')[0] for name in paired_samples['reference_file']])

    # write into: plot_binaryCE_sigma_for_CrEDC_1e4.json
    result = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
              'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
    result_as_list = {
        key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
        key, value in result.items()}
    result_as_list['failure'] = case_ids[mask].tolist()  # 'failure': case_ids[mask]
    save_json(result_as_list, join(folder_save, f'plot_{output_file}'), sort_keys=False)
    # nnUNet_results_inference/Brats2021_holdin/Dataset137_BraTS2021/2d_CE_DC/fold_0/ratio_metrics_prob/
    ################################################
    # # "plots_interval": r, r_corr
    plot_r_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10)
    plot_r_and_range_dataset(paired_samples, folder_save, sigma="", step_size=15)
    plot_r_and_range_dataset(paired_samples, folder_save, sigma=3, step_size=10)
    plot_r_and_range_dataset(paired_samples, folder_save, sigma=3, step_size=15)

    # "plots_interval_sep": ce, ce+sigma
    plot_ce_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10)
    plot_ce_and_range_dataset(paired_samples, folder_save, sigma="", step_size=15)
    plot_ce_and_range_dataset(paired_samples, folder_save, sigma=3, step_size=10)
    plot_ce_and_range_dataset(paired_samples, folder_save, sigma=3, step_size=15)

    # "bins_interval": bias_r, range
    plot_bins_dataset(paired_samples, folder_save, sigma="")
    plot_bins_dataset(paired_samples, folder_save, sigma="")
    plot_bins_dataset(paired_samples, folder_save, sigma=3)
    plot_bins_dataset(paired_samples, folder_save, sigma=3)
    ################################################
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               binary: bool = False):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']  # .nii.gz for segs
    # file_ending = ".npz" #.npz for probs

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                num_processes, chill=chill, binary=binary)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true',
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')
    parser.add_argument('--binary', action='store_true',
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, binary=args.binary)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true',
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il,
                                     chill=args.chill)


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

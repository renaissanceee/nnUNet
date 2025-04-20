import multiprocessing
import os
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
from nnunetv2.evaluation.ece_utils import fast_ece, brier_score
from nnunetv2.evaluation.ece_kde import get_ece_kde
from nnunetv2.evaluation.plot_utils import plot_r_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F


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


def analyze_r_ce_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                     tensor_seg_gt_map, ce_type):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,
                                                     n)
    var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    sigma_r = var_r ** 0.5
    #####################
    if ce_type == 'kde':
        epsilon_y, epsilon_x = calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                            tensor_wt_gt_map)
    elif ce_type == 'bins':
        epsilon_y, epsilon_x = calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                             tensor_wt_gt_map)
    elif ce_type == 'bs':
        epsilon_y, epsilon_x = calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    elif ce_type == 'nll':
        epsilon_y, epsilon_x = calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    #####################
    ce_left = y_bar / x_bar - max(y_bar - epsilon_y, 0) / (x_bar + epsilon_x)  # extreme-case: move to 0
    ce_right = (y_bar + epsilon_y) / max(x_bar - epsilon_x, 0) - y_bar / x_bar  # move to 1
    # overall-range
    l_range = ce_left + sigma_r  # CE_left + σ
    r_range = ce_right + sigma_r  # CE_right + σ    # range__ce+1std = ce_left + ce_right + 2*sigma_r
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
    # overall-range
    l_range = ce_left + sigma_r  # CE_left + σ
    r_range = ce_right + sigma_r  # CE_right + σ    # range__ce+1std = ce_left + ce_right + 2*sigma_r
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


def get_ece_bins(f, y, device):
    f, y = f.squeeze(1).to(device), y.to(device)
    batch_ratio = fast_ece(f, y, n_bins=10, device=device).to("cpu")
    return batch_ratio
    # batch_ratio = fast_ece(f, y, n_bins=10, device=device).to("cpu").item()
    # return torch.tensor(batch_ratio)


def calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing ece_kde ...")
    # 1d_kde
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    bandwidth = 0.02  # 0.001
    device = "cuda"
    sub = 1e4
    epsilon_y = get_ece_kde_sub(tensor_nec_prob_map, tensor_nec_gt_map, bandwidth, p=1,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs 2
    epsilon_x = get_ece_kde_sub(tensor_wt_prob_map.to(device), tensor_wt_gt_map.to(device), bandwidth, p=1,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs {1,2,3}
    return epsilon_y, epsilon_x


def calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing Brier_score ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    epsilon_y = brier_score(tensor_nec_prob_map.squeeze(1), tensor_nec_gt_map)
    epsilon_x = brier_score(tensor_wt_prob_map.squeeze(1), tensor_wt_gt_map)
    return epsilon_y, epsilon_x


def calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    # def calc_nll(tensor_seg_prob_map, tensor_seg_gt_map):
    print(f"Analyzing NLL ...")
    # tensor_seg_prob_map = tensor_seg_prob_map.permute(1, 2, 3, 0).reshape(-1, 4)
    # tensor_seg_gt_map = tensor_seg_gt_map.reshape(-1)
    # epsilon_y = torch.tensor(log_loss(tensor_seg_gt_map, tensor_seg_prob_map, labels=[0, 1, 2, 3]))
    # epsilon_x = epsilon_y

    # epsilon_y = log_loss(tensor_nec_gt_map.reshape(-1), tensor_nec_prob_map.reshape(-1))
    # epsilon_x = log_loss(tensor_wt_gt_map.reshape(-1), tensor_wt_prob_map.reshape(-1))
    # mask = (tensor_seg_gt_map == 1) | (tensor_seg_gt_map[:, 1] > 0)  # 可选条件：只关注与类1有关的样本
    # y_true_binary = (y_true[mask] == 1).astype(int)  # 只保留 1 是正类，其余为0
    # y_pred_binary = y_pred[mask, 1]  # 只取第1类的概率
    # epsilon_y = log_loss(y_true_binary, y_pred_binary)
    tensor_wt_prob_map = torch.clamp(tensor_wt_prob_map, min=0, max=1)
    epsilon_y = F.binary_cross_entropy(tensor_nec_prob_map.reshape(-1), tensor_nec_gt_map.reshape(-1).double(),
                                       reduction='mean')
    epsilon_x = F.binary_cross_entropy(tensor_wt_prob_map.reshape(-1), tensor_wt_gt_map.reshape(-1).double(),
                                       reduction='mean')
    return epsilon_y, epsilon_x


def calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing ece_bins ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    epsilon_y = get_ece_bins(tensor_nec_prob_map, tensor_nec_gt_map, device="cuda")  # binary: 0 vs 2
    epsilon_x = get_ece_bins(tensor_wt_prob_map, tensor_wt_gt_map, device="cuda")  # binary: 0 vs {1,2,3}
    return epsilon_y, epsilon_x


def clip_to_unit_range(x):
    return np.clip(x, 0, 1)  # range [0, 1]


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


def check_cropped_outside_is_zero(tensor, bbox):
    mask = torch.zeros_like(tensor, dtype=bool)  # 创建一个全为 False 的掩码
    slices = tuple(slice(low, high) for low, high in bbox)  # 在 bbox内置为 True
    mask[slices] = True
    outside_region = tensor[~mask]  # 检查 bbox 外是否为 0
    return torch.all(outside_region == 0)


def find_nonzero_range(tensor: torch.Tensor, cropped_z: int = 100, cropped_size: int = 190) -> torch.Tensor:
    """
    Crop a [D, H, W] tensor around the first detected non-zero column and row in the [H, W] dimensions.

    Args:
        tensor (torch.Tensor): Input tensor with shape [D, H, W]->[D, cropped_size, cropped_size]
    """
    D, H, W = tensor.shape

    # Find non-zero columns and rows
    col_has_value = (tensor != 0).any(dim=0).any(dim=0)  # [W]
    row_has_value = (tensor != 0).any(dim=0).any(dim=1)  # [H]
    z_has_value = (tensor != 0).any(dim=1).any(dim=1)  # [D]

    # Get first non-zero indices
    first_nonzero_col = torch.where(col_has_value)[0][0].item()
    first_nonzero_row = torch.where(row_has_value)[0][0].item()
    first_nonzero_z = torch.where(z_has_value)[0][0].item()  # z_min

    # Prevent out-of-bounds crop
    first_nonzero_z = D - cropped_z if first_nonzero_z + cropped_z > D else first_nonzero_z
    first_nonzero_col = W - cropped_size if first_nonzero_col + cropped_size > W else first_nonzero_col
    first_nonzero_row = H - cropped_size if first_nonzero_row + cropped_size > H else first_nonzero_row

    if cropped_z == 155:
        first_nonzero_z = 0
    return first_nonzero_z, first_nonzero_col, first_nonzero_row  # W,H


def detect_failure(paired_samples):  # fail_case: outside the range
    fail_case = []
    r_gt = paired_samples['r_gt']['r_gt']
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+1std'][:, 0], paired_samples['r_naive'][
                                                                                  'bound__ce+1std'][:, 1]
    mask_1 = (r_gt < lower_std) | (r_gt > upper_std)
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+2std'][:, 0], paired_samples['r_naive'][
                                                                                  'bound__ce+2std'][:, 1]
    mask_2 = (r_gt < lower_std) | (r_gt > upper_std)
    lower_std, upper_std = paired_samples['r_naive']['bound__ce+3std'][:, 0], paired_samples['r_naive'][
                                                                                  'bound__ce+3std'][:, 1]
    mask_3 = (r_gt < lower_std) | (r_gt > upper_std)
    case_ids = np.array(
        [os.path.basename(name).split('_')[-1].split('.')[0] for name in paired_samples['reference_file']])
    return case_ids[mask_1], case_ids[mask_2], case_ids[mask_3]


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      cropped_size: int = None,
                      cropped_z: int = None,
                      ce_type: str = "bins") -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, f'epsilon_ece_{ce_type}': {}, 'r_naive': {}, 'r_first_corr': {},
                        'r_second_corr': {},
                        'iou_scores': {}}
    print(f"calculate r for {os.path.basename(reference_file)}")

    "analyze mean/var"
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, (2,))

    # print(np.sum(slice_data== 2),np.sum(slice_data== 1)+np.sum(slice_data== 2)+np.sum(slice_data== 3))
    # pred
    wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
    nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (2,))  # numerator

    # prob+gt
    tensor_nec_prob_map = torch.from_numpy(nec_prob_map)  # [155,240,240]
    tensor_wt_prob_map = torch.from_numpy(wt_prob_map)
    tensor_nec_gt_map = torch.from_numpy(nec_gt_map).squeeze(0)
    tensor_wt_gt_map = torch.from_numpy(wt_gt_map).squeeze(0)

    tensor_seg_prob_map = torch.from_numpy(prob_pred)
    tensor_seg_gt_map = torch.from_numpy(seg_ref)
    # torch.clamp(f[idx, :], min=0, max=1)
    # binary
    if binary:
        print("Hey, now we binarize preds for seg (not prob)")
        tensor_seg_map = torch.from_numpy(seg_pred)  # seg: argmax for 1
        tensor_nec_prob_map = (tensor_seg_map == 2).to(torch.float64)
        tensor_wt_prob_map = torch.isin(tensor_seg_map, torch.tensor([1, 2, 3])).double()

    ## crop for smaller N
    if cropped_size is not None:  # bbox = [[0, 155], [25, 215], [25, 215]]
        # [155,240,240]->[140,190,190]
        first_wt_z, first_wt_col, first_wt_row = find_nonzero_range(tensor_wt_prob_map, cropped_z, cropped_size)
        bbox = [[first_wt_z, first_wt_z + cropped_z], [first_wt_row, first_wt_row + cropped_size],
                [first_wt_col, first_wt_col + cropped_size]]
        if not check_cropped_outside_is_zero(tensor_wt_gt_map, bbox):
            assert False, f"'outside has tumor!"
        else:
            print(f"Cropping ... to {bbox}")
        tensor_nec_gt_map, tensor_nec_prob_map = crop_to_bbox(tensor_nec_gt_map, bbox), crop_to_bbox(
            tensor_nec_prob_map, bbox)
        tensor_wt_gt_map, tensor_wt_prob_map = crop_to_bbox(tensor_wt_gt_map, bbox), crop_to_bbox(tensor_wt_prob_map,
                                                                                                  bbox)
        # count_ones = (outside_region == 1).sum().item()

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
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] = epsilon_y
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] = epsilon_x
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


def compute_estimator_avg(reference_file: str, prediction_file: str, probability_file: str,
                          image_reader_writer: BaseReaderWriter,
                          labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                          ignore_label: int = None,
                          binary: bool = False,
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
    results['ratio'] = {'r_gt': {}, f'epsilon_ece_{ce_type}': {}, 'r_naive': {}, 'r_first_corr': {},
                        'r_second_corr': {},
                        'iou_scores': {}}
    print(f"calculate r for {os.path.basename(reference_file)}")

    "analyze mean/var"
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, (2,))

    # print(np.sum(slice_data== 2),np.sum(slice_data== 1)+np.sum(slice_data== 2)+np.sum(slice_data== 3))
    # pred
    wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
    nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (2,))  # numerator

    # prob+gt
    tensor_nec_prob_map = torch.from_numpy(nec_prob_map)  # [155,240,240]
    tensor_wt_prob_map = torch.from_numpy(wt_prob_map)
    tensor_nec_gt_map = torch.from_numpy(nec_gt_map).squeeze(0)
    tensor_wt_gt_map = torch.from_numpy(wt_gt_map).squeeze(0)

    tensor_seg_prob_map = torch.from_numpy(prob_pred)
    tensor_seg_gt_map = torch.from_numpy(seg_ref)

    # binary
    if binary:
        print("Hey, now we binarize preds for seg (not prob)")
        tensor_seg_map = torch.from_numpy(seg_pred)  # seg: argmax for 1
        tensor_nec_prob_map = (tensor_seg_map == 2).to(torch.float64)
        tensor_wt_prob_map = torch.isin(tensor_seg_map, torch.tensor([1, 2, 3])).double()

    "reformulate r to see how interval(x,y) changes"
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
    # CE: y,x
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] = epsilon_y
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] = epsilon_x
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
                                binary: bool = False,
                                cropped_size: int = None,
                                cropped_z: int = None,
                                ce_type: str = "bins",  # "kde", nll, bs
                                exp: int = None,  # diff exp for kde,
                                ) -> dict:
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
    results = []
    # i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        # i += 1
        # if i > 130 or i < 128: continue  # JJ: first two samples
        # if ref!="/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/BraTS2021_00218.nii.gz": continue
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, cropped_size, cropped_z, ce_type)
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
    paired_samples[f'epsilon_ece_{ce_type}'] = {}
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] for item in results])
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] for item in results])
    mean_epsilon['epsilon_y'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'])
    failure_1std, failure_2std, failure_3std = detect_failure(paired_samples)  # fail_case: outside the range
    failure = {}
    failure['std'], failure['2std'], failure['3std'] = failure_1std, failure_2std, failure_3std
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_jaccard, mean_r_range, mean_r_bias, mean_epsilon, failure]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_r_bias': mean_r_bias,
              'mean_r_range__ce+123std': mean_r_range,
              'failure': failure,
              'mean_r_jaccard__ce+123std': mean_r_jaccard,
              'ratio_per_case': results,
              }

    folder_save = join(folder_pred, "ratio_metrics_binary") if binary else join(folder_pred,
                                                                                "ratio_metrics_prob")  # binary
    folder_save = f"{folder_save}_crop{cropped_size}&{cropped_z}" if cropped_size is not None else folder_save  # crop
    os.makedirs(folder_save, exist_ok=True)
    save_summary_json(result, join(folder_save, f"{ce_type}_{output_file}"))

    result_plot = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
                   'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
    result_as_list = {
        key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
        key, value in result_plot.items()}  # extract bound-related items
    save_json(result_as_list, join(folder_save, f"plot_{ce_type}_{output_file}"), sort_keys=False)
    ################################################
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
                                    ce_type: str = "kde",  # "kde", nll, bs
                                    exp: str = "avg",  # diff exp for kde,
                                    avg_epsilon_y: Any = None,
                                    avg_epsilon_x: Any = None) -> dict:
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
    results = []

    for ref, pred, prob, epsilon_y, epsilon_x in zip(files_ref, files_pred, files_prob, avg_epsilon_y, avg_epsilon_x):
    # for ref, pred, prob, epsilon_y, epsilon_x in zip(files_ref[:3], files_pred[:3], files_prob[:3], avg_epsilon_y[:3], avg_epsilon_x[:]):
        result = compute_estimator_avg(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                       binary, ce_type, epsilon_y, epsilon_x)
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
    paired_samples[f'epsilon_ece_{ce_type}'] = {}
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] for item in results])
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] for item in results])
    mean_epsilon['epsilon_y'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'])
    failure_1std, failure_2std, failure_3std = detect_failure(paired_samples)  # fail_case: outside the range
    failure = {}
    failure['std'], failure['2std'], failure['3std'] = failure_1std, failure_2std, failure_3std
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_jaccard, mean_r_range, mean_r_bias, mean_epsilon, failure]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_r_bias': mean_r_bias,
              'mean_r_range__ce+123std': mean_r_range,
              'failure': failure,
              'mean_r_jaccard__ce+123std': mean_r_jaccard,
              'ratio_per_case': results,
              }

    folder_save = join(folder_pred, "ratio_metrics_binary") if binary else join(folder_pred,
                                                                                "ratio_metrics_prob")  # binary
    os.makedirs(folder_save, exist_ok=True)
    save_summary_json(result, join(folder_save, f"{ce_type}_{output_file}"))

    result = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
              'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
    result_as_list = {
        key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
        key, value in result.items()}  # extract bound-related items

    failure_1std, failure_2std, failure_3std = detect_failure(paired_samples)  # fail_case: outside the range

    save_json(result_as_list, join(folder_save, f"plot_{ce_type}_{output_file}"), sort_keys=False)
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
                               cropped_size: int = 0,
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
    cropped_z = 140 if cropped_size is not None else None  # fixed num_slices when cropping

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    repeat_for_kde = 5

    if ce_type == 'kde':
        ## repeate 5 times
        list_epsilon_y, list_epsilon_x = [], []
        for exp in range(repeat_for_kde):
            print(f'JJ: repeate on time {exp}...')
            output_file_exp = output_file.replace('.json', f"_1e4_{exp}.json")  # 1e4
            ##  ['epsilon_ece_kde']['epsilon_y'] and 'x'  for 2 np.arrays
            epsilon_dict = compute_estimator_on_folder(folder_ref, folder_pred, output_file_exp, rw, file_ending,
                                                       lm.foreground_regions if lm.has_regions else lm.foreground_labels,
                                                       lm.ignore_label,
                                                       num_processes, chill=chill, binary=binary,
                                                       cropped_size=cropped_size, cropped_z=cropped_z, ce_type='kde',
                                                       exp=exp)
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
                                        ce_type='kde', exp='avg',
                                        avg_epsilon_y=avg_epsilon_y, avg_epsilon_x=avg_epsilon_x)

        # ## update 'epsilon_ece_kde'
        # [dict.update(result['ratio_per_case'][i]["ratio"]["epsilon_ece_kde"], {"epsilon_y": avg_epsilon_y[i]}) for i in range(len(result['ratio_per_case']))]
        # [dict.update(result['ratio_per_case'][i]["ratio"]["epsilon_ece_kde"], {"epsilon_x": avg_epsilon_x[i]}) for i in range(len(result['ratio_per_case']))]
        # ## update 'mean_ece_kde'
        # result['mean_ece_kde'] = {"epsilon_y": np.mean(avg_epsilon_y), "epsilon_x": np.mean(avg_epsilon_x)}




    else:
        compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                    lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                    num_processes, chill=chill, binary=binary, cropped_size=cropped_size,
                                    cropped_z=cropped_z, ce_type=ce_type, exp=None)


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
        output_file = join(folder_pred, 'ratio.json')
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
    parser.add_argument('--binary', action='store_true', help='use seg instead of prob for ratio est')
    parser.add_argument('--TS', type=str, required=False, default=None, help='Temperature Scaling')
    parser.add_argument('--crop', type=int, required=False, default=None, help='crop for smaller N')
    parser.add_argument('--ce_type', required=True, type=str, help='bins, kde, nll, bs')
    args = parser.parse_args()
    if args.TS is not None:
        basename = os.path.basename(args.pred_folder)
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_TS_{args.TS}')
        print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, binary=args.binary, cropped_size=args.crop, ce_type=args.ce_type)


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
    parser.add_argument('--binary', action='store_true', help='use seg instead of prob for ratio est')
    parser.add_argument('--TS', type=str, required=False, default=None, help='Temperature Scaling')
    parser.add_argument('--crop', type=int, required=False, default=None, help='crop for smaller N')
    args = parser.parse_args()
    if args.TS is not None:
        basename = os.path.basename(args.pred_folder)
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_TS_{args.TS}')
        print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il,
                                     chill=args.chill, binary=args.binary, cropped_size=args.crop)


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

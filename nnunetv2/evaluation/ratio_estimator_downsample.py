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
from nnunetv2.evaluation.ece_utils import fast_ece, brier_score, mean_lp_dist
from nnunetv2.evaluation.ece_kde import get_ece_kde
from nnunetv2.evaluation.crop_utils import check_cropped_outside_is_zero,find_nonzero_range, downsample_3d_tensor
from nnunetv2.evaluation.plot_utils import plot_r_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
import random


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

def analyze_r_std(tensor_nec_prob_map, tensor_wt_prob_map):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,n)
    var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    sigma_r = var_r ** 0.5
    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), sigma_r.item()


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
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1), tensor_wt_prob_map.reshape(-1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(-1).to(torch.int64)
    epsilon_y = brier_score(tensor_nec_prob_map, tensor_nec_gt_map)
    epsilon_x = brier_score(tensor_wt_prob_map, tensor_wt_gt_map)
    mean_l2_y = mean_lp_dist(tensor_nec_prob_map, tensor_nec_gt_map, p=2)
    mean_l2_x = mean_lp_dist(tensor_wt_prob_map, tensor_wt_gt_map, p=2)
    return epsilon_y, epsilon_x, mean_l2_y, mean_l2_x


def calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    print(f"Analyzing NLL ...")
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

def update_r_keys(results, key, r_naive, r_gt):
    results['ratio'][key]['r_est'] = r_naive
    results['ratio'][key]['r_bias'] = r_naive - r_gt
    return results


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      biomarker: str ='ntr',
                      ce_type: str = "bins",
                      sampling_factor: int = 2) -> dict:
    cropped_z, cropped_size = None, None
    # cropped_z, cropped_size = 140, 190

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


    if cropped_z is not None:  # bbox = [[0, 155], [25, 215], [25, 215]]
        # [155,240,240]->[140,190,190]
        first_wt_z, first_wt_col, first_wt_row = find_nonzero_range(tensor_wt_prob_map, cropped_z, cropped_size)
        bbox = [[first_wt_z, first_wt_z + cropped_z], [first_wt_row, first_wt_row + cropped_size],
                [first_wt_col, first_wt_col + cropped_size]]
        if not check_cropped_outside_is_zero(tensor_wt_gt_map, bbox):
            assert False, f"'outside has tumor!"
        # print(f"Cropping ... to {bbox}")
        tensor_nec_gt_map, tensor_nec_prob_map = crop_to_bbox(tensor_nec_gt_map, bbox), crop_to_bbox(
            tensor_nec_prob_map, bbox)
        tensor_wt_gt_map, tensor_wt_prob_map = crop_to_bbox(tensor_wt_gt_map, bbox), crop_to_bbox(tensor_wt_prob_map,bbox)
    # large N for r_gt
    r_gt = nec_gt_counter / wt_gt_counter
    ################################
    ## downsample + scattor
    sample_times = 10
    seeds = np.arange(1, sample_times, dtype=int)
    r_naive_list, r_1_ord_corr_list, r_2_ord_corr_list, sigma_r_list = [],[],[],[]
    for seed in seeds:
        # downsample: no change distr
        down_nec_prob_map = downsample_3d_tensor(tensor_nec_prob_map, sampling_factor, seed)# [155,240,240]->[77,120,120]
        down_wt_prob_map = downsample_3d_tensor(tensor_wt_prob_map, sampling_factor, seed)
        down_nec_gt_map = downsample_3d_tensor(tensor_nec_gt_map, sampling_factor, seed)
        down_wt_gt_map = downsample_3d_tensor(tensor_wt_gt_map, sampling_factor, seed)
        # r_naive, r_1_ord_corr, r_2_ord_corr
        y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(down_nec_prob_map,down_wt_prob_map)
        r_naive, r_1_ord_corr, r_2_ord_corr, sigma_r = analyze_r_std(down_nec_prob_map,down_wt_prob_map)
        # append
        r_naive_list.append(r_naive)
        r_1_ord_corr_list.append(r_1_ord_corr)
        r_2_ord_corr_list.append(r_2_ord_corr)
        sigma_r_list.append(sigma_r)
        # print("naive(large) ", np.abs(r_naive - r_gt))
        # print("2-ord(small) ", np.abs(r_2_ord_corr - r_gt))
        # print('------------------')

    r_naive = np.mean(r_naive_list)
    r_1_ord_corr = np.mean(r_1_ord_corr_list)
    r_2_ord_corr = np.mean(r_2_ord_corr_list)
    sigma_r = np.mean(sigma_r_list)
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {},'r_naive': {}, 'r_first_corr': {},'r_second_corr': {}}
    # gt
    results['ratio']['r_gt']['r_gt'] = r_gt
    results['ratio']['r_gt']['sigma_r'] = sigma_r
    # estimation
    update_r_keys(results, 'r_naive', r_naive, r_gt)
    update_r_keys(results, 'r_first_corr', r_1_ord_corr, r_gt)
    update_r_keys(results, 'r_second_corr', r_2_ord_corr, r_gt)

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
                                sampling_factor: int = 2,
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
        # if i > 130 or i < 129: continue  # JJ: first two samples
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, biomarker, ce_type, sampling_factor)
        results.append(result)


    ################################################
    # r_gt and ref.nii.gz
    paired_samples = {'r_gt':{}, 'r_naive':{}, 'r_first_corr':{}, 'r_second_corr':{}}
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    mean_r_bias = {}
    scores = ['r_naive', 'r_first_corr', 'r_second_corr']
    for r in scores:
        paired_samples[r]['bias_r'] = np.array([item['ratio'][r]['r_bias'] for item in results])
        paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
        mean_r_bias[r] = np.mean(paired_samples[r]['bias_r'])
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_bias]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {'mean_r_bias': mean_r_bias,
              'ratio_per_case': results,
              }

    folder_save = join(folder_pred, "ratio_metrics_binary") if binary else join(folder_pred,"ratio_metrics_prob")  # binary
    # folder_save = join(folder_save,'downsample')
    folder_save = join(folder_save, f'downsample_{biomarker}')
    os.makedirs(folder_save, exist_ok=True)
    save_json(result, join(folder_save, f"down_{sampling_factor}_{output_file}"), sort_keys=False)

    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               binary: bool = False,
                               biomarker: str ='ntr',
                               ce_type: str = "bins",  # "kde", nll, bs,
                               sampling_factor: int = 2,
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
    compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                num_processes, chill=chill, binary=binary, biomarker=biomarker,ce_type=ce_type, sampling_factor=sampling_factor)


# def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
#                                      output_file: str = None,
#                                      num_processes: int = default_num_processes,
#                                      ignore_label: int = None,
#                                      chill: bool = False):
#     example_file = subfiles(folder_ref, join=True)[0]
#     file_ending = os.path.splitext(example_file)[-1]
#     rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
#                                                   verbose=False)()
#     # maybe auto set output file
#     if output_file is None:
#         output_file = join(folder_pred, 'ratio.json')
#     compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
#                               labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


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
    parser.add_argument('--crop', type=int, required=False, default=None, help='crop for smaller N')
    parser.add_argument('--ce_type', required=True, type=str, help='bins, kde, nll, bs')
    parser.add_argument('--biomarker', required=True, type=str, help='ntr, ctr')
    parser.add_argument('--sampling_factor', type=int, required=True, default=2, help='downsample for smaller N')
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
                               chill=args.chill, binary=args.binary, biomarker=args.biomarker, ce_type=args.ce_type, sampling_factor=args.sampling_factor)


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

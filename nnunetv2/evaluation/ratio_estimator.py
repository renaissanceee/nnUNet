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
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import torch
from nnunetv2.evaluation.ece_utils import calc_ece_kde,calc_bs,calc_v_bias,calc_nll,calc_ece_bins
from nnunetv2.evaluation.plot_utils import plot_corr_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset,\
                                            plot_ratio_and_range_dataset, plot_ratio_and_range_all
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from nnunetv2.evaluation.seg_utils import gather_files, region_or_label_to_mask, region_or_label_to_mask_prob_add,\
    compute_tp_fp_fn_tn,get_ce_bound


def analyze_r_ce_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                     tensor_seg_gt_map, ce_type, epsilon_y_mean_ece, epsilon_x_mean_ece):
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
    elif ce_type == 'nll':
        epsilon_y, epsilon_x = calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    #####################
    if epsilon_y_mean_ece is not None and ce_type!='bs' and ce_type!='kde2': # kde/bins in test-set
        ce_left, ce_right = get_ce_bound(y_bar, x_bar, epsilon_y_mean_ece, epsilon_x_mean_ece)# ce_bound
        ce_left, ce_right = ce_left.item(), ce_right.item()
    else:
        ce_left, ce_right = 0, 0
    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), ce_left, ce_right, sigma_r.item(), epsilon_y.item(), epsilon_x.item()


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


def clip_to_unit_range(x, y):# range [0, 1]
    return np.clip(x, 0, 1),np.clip(y, 0, 1)

def generate_confidence_bounds(r, ce_left, ce_right, sigma_r):
    """
    Generate CE, CE+1σ, CE+2σ, CE+3σ intervals.
    Returns: (x_ce, y_ce), (x1σ, y1σ), (x2σ, y2σ), (x3σ, y3σ)
    """
    x_ce, y_ce = clip_to_unit_range(r - ce_left, r + ce_right)
    x_1std, y_1std = clip_to_unit_range(x_ce - sigma_r, y_ce + sigma_r)
    x_2std, y_2std = clip_to_unit_range(x_1std - sigma_r, y_1std + sigma_r)
    x_3std, y_3std = clip_to_unit_range(x_1std - 2 * sigma_r, y_1std + 2 * sigma_r)
    return (x_ce, y_ce), (x_1std, y_1std), (x_2std, y_2std), (x_3std, y_3std)

def detect_failure(paired_samples):
    """Detect failure cases: r_gt falls outside of CE + Nσ bounds."""
    r_gt = paired_samples['r_gt']['r_gt']
    case_ids = np.array([
        os.path.basename(name).split('_')[-1].split('.')[0]
        for name in paired_samples['reference_file']
    ])

    def out_of_bound(mask_key):
        bounds = paired_samples['r_naive'][mask_key]
        lower, upper = bounds[:, 0], bounds[:, 1]
        return (r_gt < lower) | (r_gt > upper)

    return case_ids[out_of_bound('bound__ce+1std')], \
           case_ids[out_of_bound('bound__ce+2std')], \
           case_ids[out_of_bound('bound__ce+3std')]


def update_r_keys_zero(results):
    """Zero out all r-related estimates when CE info is not available."""
    for key in ['r_naive', 'r_first_corr', 'r_second_corr']:
        results['ratio'][key] = {
            'r_est': 0,
            'r_bias': 0,
            'bound__ce': np.array([0, 0]),
            'bound__ce+1std': np.array([0, 0]),
            'bound__ce+2std': np.array([0, 0]),
            'bound__ce+3std': np.array([0, 0]),
            'range__ce+1std': 0,
            'range__ce+2std': 0,
            'range__ce+3std': 0,
        }
    return results

def update_r_keys(results, key, r_est, r_gt, *bounds):
    (x_ce, y_ce, x_1std, y_1std, x_2std, y_2std, x_3std, y_3std) = bounds
    results['ratio'][key] = {
        'r_est': r_est,
        'r_bias': r_est - r_gt,
        'bound__ce': np.array([x_ce, y_ce]),
        'bound__ce+1std': np.array([x_1std, y_1std]),
        'bound__ce+2std': np.array([x_2std, y_2std]),
        'bound__ce+3std': np.array([x_3std, y_3std]),
        'range__ce+1std': y_1std - x_1std,
        'range__ce+2std': y_2std - x_2std,
        'range__ce+3std': y_3std - x_3std,
    }
    return results

def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      biomarker: str ='ntr',
                      ce_type: str = "bins",
                      epsilon_y_mean_ece: float = None,
                      epsilon_x_mean_ece: float = None,
                      ) -> dict:
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
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,tensor_wt_prob_map)
    # calib-error（epsilon_y, epsilon_x）
    r_naive, r_1_ord_corr, r_2_ord_corr, ce_left, ce_right, sigma_r, epsilon_y, epsilon_x = analyze_r_ce_std(
        tensor_nec_prob_map,
        tensor_wt_prob_map,
        tensor_nec_gt_map,
        tensor_wt_gt_map,
        tensor_seg_prob_map,
        tensor_seg_gt_map,
        ce_type,
        epsilon_y_mean_ece, # for bound
        epsilon_x_mean_ece)
    # print(f'ce_l: {ce_left}, ce_r: {ce_right}') # 0.109, 0.162
    # print(f'std: {sigma_r}') # 0.001
    # CE_range
    # a prior is: r>0, which is appicable for confidence range
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'v_bias': {}, f'epsilon_ece_{ce_type}': {},
                        'r_naive': {}, 'r_first_corr': {},'r_second_corr': {}}
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

    if epsilon_y_mean_ece is not None and ce_type!='bs' and ce_type!='kde2':
        bounds_naive = generate_confidence_bounds(r_naive, ce_left, ce_right, sigma_r)
        bounds_1corr = generate_confidence_bounds(r_1_ord_corr, ce_left, ce_right, sigma_r)
        bounds_2corr = generate_confidence_bounds(r_2_ord_corr, ce_left, ce_right, sigma_r)

        update_r_keys(results, 'r_naive', r_naive, r_gt, *sum(bounds_naive, ()))
        update_r_keys(results, 'r_first_corr', r_1_ord_corr, r_gt, *sum(bounds_1corr, ()))
        update_r_keys(results, 'r_second_corr', r_2_ord_corr, r_gt, *sum(bounds_2corr, ()))

    else:
        update_r_keys_zero(results)


    return results


def compute_estimator_avg(reference_file: str, prediction_file: str, probability_file: str,
                          image_reader_writer: BaseReaderWriter,
                          labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                          ignore_label: int = None,
                          binary: bool = False,
                          biomarker: str ='ntr',
                          ce_type: str = "kde",
                          epsilon_y_mean_ece=None, epsilon_x_mean_ece=None) -> dict:
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
                        'r_naive': {}, 'r_first_corr': {},'r_second_corr': {}}
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
        epsilon_y_mean_ece, # for bound
        epsilon_x_mean_ece)

    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'v_bias': {}, f'epsilon_ece_{ce_type}': {},
                        'r_naive': {}, 'r_first_corr': {},'r_second_corr': {}}
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

    if epsilon_y_mean_ece is not None:
        bounds_naive = generate_confidence_bounds(r_naive, ce_left, ce_right, sigma_r)
        bounds_1corr = generate_confidence_bounds(r_1_ord_corr, ce_left, ce_right, sigma_r)
        bounds_2corr = generate_confidence_bounds(r_2_ord_corr, ce_left, ce_right, sigma_r)

        update_r_keys(results, 'r_naive', r_naive, r_gt, *sum(bounds_naive, ()))
        update_r_keys(results, 'r_first_corr', r_1_ord_corr, r_gt, *sum(bounds_1corr, ()))
        update_r_keys(results, 'r_second_corr', r_2_ord_corr, r_gt, *sum(bounds_2corr, ()))
    else:
        update_r_keys_zero(results)
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
    if binary:
        folder_save = join(folder_pred, f"ratio_metrics_binary_{biomarker}")
    else:
        folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
    os.makedirs(folder_save, exist_ok=True)
    if 'test' in folder_pred and ce_type!='bs' and ce_type!='kde2':
        path_ece_json = join(folder_save.replace('test', 'validation_ece'), f"{ce_type}_{output_file}")
        if 'binary' in path_ece_json:
            path_ece_json=path_ece_json.replace('binary','prob') # ECE_interval still from prob
        print(f'loading epsilon from {path_ece_json}') # bins15_ratio.json
        mean_ece = load_json(path_ece_json)
        epsilon_y_mean_ece = mean_ece[f'mean_ece_{ce_type}']['epsilon_y']
        epsilon_x_mean_ece = mean_ece[f'mean_ece_{ce_type}']['epsilon_x']
    else:
    # elif 'validation' in folder_pred:
        epsilon_y_mean_ece, epsilon_x_mean_ece = None, None
    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    ################################################
    i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob, ):
        # i += 1
        # if i > 10 or i < 8: continue  # JJ: first two samples
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, biomarker, ce_type, epsilon_y_mean_ece, epsilon_x_mean_ece)
        results.append(result)

    ################################################
    paired_samples = {'r_gt':{},'r_naive':{}, 'r_first_corr':{}, 'r_second_corr':{},
                      f'epsilon_ece_{ce_type}':{},'v_bias':{}, 'naive_vs_second__ce+123std':{}}
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
    if epsilon_y_mean_ece is not None:
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
        head_results = [mean_r_range, mean_r_bias, mean_epsilon, mean_v_bias, failure]
        [recursive_fix_for_json_export(i) for i in head_results]

        result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_v_bias': mean_v_bias,
                  'mean_r_bias': mean_r_bias,
                  'mean_r_range__ce+123std': mean_r_range,
                  'failure': failure,
                  'ratio_per_case': results,
                  }

        ## ratio.json
        save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)
        # plot.json
        result_plot = {'r_gt': paired_samples['r_gt'], 'r_naive': paired_samples['r_naive'],
                       'r_first_corr': paired_samples['r_first_corr'], 'r_second_corr': paired_samples['r_second_corr']}
        result_as_list = {
            key: {subkey: value.tolist() if isinstance(value, np.ndarray) else value for subkey, value in value.items()} for
            key, value in result_plot.items()}  # extract bound-related items
        os.makedirs(join(folder_save,'plot'), exist_ok=True)
        save_json(result_as_list, join(folder_save,'plot', f"plot_{ce_type}_{output_file}"), sort_keys=False)
        # plot figures
        sigmas = ['', 2, 3]
        step_size = 10
        for sigma in sigmas:
            plot_corr_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # r, r_corr
            plot_ce_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, exp)  # ce, ce+sigma
            plot_bins_dataset(paired_samples, folder_save, sigma, ce_type, exp)  # "hist_of bias_and_range": bias_r, range
            plot_ratio_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type)
            plot_ratio_and_range_all(paired_samples, folder_save, sigma, ce_type)
    else:
        [recursive_fix_for_json_export(i) for i in results]
        recursive_fix_for_json_export(mean_epsilon)
        result = {f'mean_ece_{ce_type}': mean_epsilon,
                  'ratio_per_case': results,
                  }
        ## ratio.json
        save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)

    return paired_samples[f'epsilon_ece_{ce_type}']  # ['epsilon_ece_kde']['epsilon_y'] and 'x' for 2 np.arrays


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

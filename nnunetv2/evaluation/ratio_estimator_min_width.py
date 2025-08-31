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
from nnunetv2.evaluation.ece_utils import calc_ece_kde, calc_bs, calc_v_bias, calc_nll, calc_ece_bins,ece_loss
from nnunetv2.evaluation.ece_utils import detect_failure, detect_failure_size, cover_rate
# from nnunetv2.evaluation.plot_utils import plot_bins_dataset, plot_ratio_and_range_all
# from nnunetv2.evaluation.plot_utils import plot_corr_and_range_dataset,
from nnunetv2.evaluation.plot_utils import plot_ce_and_range_dataset, plot_ratio_and_range_dataset, plot_histogram_range
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from nnunetv2.evaluation.seg_utils import gather_files, region_or_label_to_mask, region_or_label_to_mask_prob_add, \
    compute_tp_fp_fn_tn, get_ce_bound, convert_labels_to_one_hot, cat_all_source_files


def analyze_r_ce_std(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                     tensor_seg_gt_map, ce_type, epsilon_y_mean_ece, epsilon_x_mean_ece, alpha, source_dict):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, r_1_ord_corr, r_2_ord_corr = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,n)
    # var_r = (y_bar ** 2 / x_bar ** 4 * var_x + var_y / x_bar ** 2 - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    mse = (y_bar / x_bar ** 4 * var_x + var_y / x_bar - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    beta = (mse / alpha) ** 0.5
    #####################
    if 'kde' in ce_type:
        p = int(re.search(r'\d+', ce_type).group())
        repeat_for_kde = 5  ## ece_kde: repeate 5 times
        list_epsilon_y, list_epsilon_x = [], []
        for i in range(repeat_for_kde):
            # print(f'JJ: repeate on time {i}...')
            epsilon_y_sub, epsilon_x_sub = calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                                        tensor_wt_gt_map, p)
            list_epsilon_y.append(epsilon_y_sub)
            list_epsilon_x.append(epsilon_x_sub)
        # epsilon_y, epsilon_x = np.mean(list_epsilon_y, axis=0), np.mean(list_epsilon_x, axis=0)
        epsilon_y, epsilon_x = torch.mean(torch.tensor(list_epsilon_y)), torch.mean(torch.tensor(list_epsilon_x))
    elif 'bins' in ce_type:
        bins = int(re.search(r'\d+', ce_type).group())
        epsilon_y, epsilon_x = calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                             tensor_wt_gt_map, bins)
    elif ce_type == 'bs':
        epsilon_y, epsilon_x = calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    elif ce_type == 'nll':
        epsilon_y, epsilon_x = calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    else:
        print("You don't specify ECE type, the use bins by default")
        bins = int(re.search(r'\d+', ce_type).group())
        epsilon_y, epsilon_x = calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                             tensor_wt_gt_map, bins)
    #####################
    # if epsilon_y_mean_ece is not None and ce_type!='bs' and ce_type!='kde2': # kde/bins in test-set
    if epsilon_y_mean_ece is not None:
        ce_left, ce_right = get_ce_bound(y_bar, x_bar, epsilon_y_mean_ece, epsilon_x_mean_ece)  # ce_bound
        ce_left, ce_right = ce_left.item(), ce_right.item()
    else:
        ce_left, ce_right = 0, 0
    return r_naive.item(), r_1_ord_corr.item(), r_2_ord_corr.item(), ce_left, ce_right, beta.item(), epsilon_y.item(), epsilon_x.item()


def analyze_r_ce_std_alpha(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, tensor_seg_prob_map,
                     tensor_seg_gt_map, ce_type, epsilon_y_mean_ece, epsilon_x_mean_ece, total_out_prob):
    # r and std
    y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                          tensor_wt_prob_map)
    r_naive, _, _ = estimate_r(y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y,n)
    mse = (y_bar / x_bar ** 4 * var_x + var_y / x_bar - 2 * y_bar / x_bar ** 3 * cov_x_y) / n
    mse = mse.item()
    y_bar, x_bar = y_bar.item(), x_bar.item()
    # total_out_prob = 0.68
    #####################
    min_width, opt_Q = 1e7, 1  # 初始化为无穷大
    opt_ce_left, opt_ce_right = None, None
    eps = 1e-7
    # for Q in np.linspace((total_out_prob+1+eps)/2, 1, num=20): ## May
    #     alpha = 2*Q - total_out_prob - 1

    for Q in np.linspace(total_out_prob+eps, 1, num=20): ## 1q
        alpha = Q - total_out_prob
        # import pdb;pdb.set_trace()

        beta = (mse / alpha) ** 0.5
        epsilon_y_q = np.quantile(epsilon_y_mean_ece, Q)
        epsilon_x_q = np.quantile(epsilon_x_mean_ece, Q)

        ce_left, ce_right = get_ce_bound(y_bar, x_bar, epsilon_y_q, epsilon_x_q)

        cur_width = ce_right + ce_left + 2 * beta

        if cur_width < min_width:
            min_width = cur_width
            opt_Q = Q
            opt_ce_left, opt_ce_right = ce_left, ce_right
            opt_alpha = alpha
    #####################
    # print("Q: ", opt_Q)
    # print("alpha: ",opt_alpha)
    # print("min_width: ", opt_ce_right+opt_ce_left)
    #####################
    return r_naive.item(), opt_ce_left, opt_ce_right, beta, opt_Q, opt_alpha

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


def clip_to_unit_range(x, y):  # range [0, 1]
    return np.clip(x, 0, 1), np.clip(y, 0, 1)


def generate_confidence_bounds(r, ce_left, ce_right, beta):
    """
    Generate CE, CE+1σ, CE+2σ, CE+3σ intervals.
    Returns: (x_ce, y_ce), (x1σ, y1σ), (x2σ, y2σ), (x3σ, y3σ)
    """
    x_ce, y_ce = clip_to_unit_range(r - ce_left, r + ce_right)
    x_1std, y_1std = clip_to_unit_range(x_ce - beta, y_ce + beta)
    return (x_ce, y_ce),(x_1std, y_1std)


def s_m_l(array):
    s_threshold, m_threshold = np.quantile(array, 0.33), np.quantile(array, 0.66)
    # print("S/M/L:", s_threshold, m_threshold, "...")
    s_mask = array <= s_threshold
    m_mask = (array > s_threshold) & (array <= m_threshold)
    l_mask = array > m_threshold
    return s_mask, m_mask, l_mask

def update_r_keys_zero(results):
    """Zero out all r-related estimates when CE info is not available."""
    for key in ['r_naive']:
        results['ratio'][key] = {
            'r_est': 0,
            'AE_r': 0,
            'bound__ce': np.array([0, 0]),
            'bound__ce+1std': np.array([0, 0]),
            'range__ce+1std': 0,
        }
    return results


def update_r_keys(results, key, r_est, r_gt, opt_Q, *bounds):
    (x_ce, y_ce, x_1std, y_1std) = bounds
    results['ratio'][key] = {
        'opt_Q': opt_Q,
        'r_est': r_est,
        'AE_r': abs(r_est - r_gt),
        'bound__ce': np.array([x_ce, y_ce]),
        'bound__ce+1std': np.array([x_1std, y_1std]),
        'range__ce+1std': y_1std - x_1std,
    }
    return results


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      biomarker: str = 'ntr',
                      ce_type: str = "bins",
                      epsilon_y_mean_ece: float = None,
                      epsilon_x_mean_ece: float = None,
                      alpha: float = None,
                      source_dict: dict = None,
                      total_out_prob: float = None,
                      swin: bool = False,
                      ) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
    # cc=np.sum(prob_pred,axis=0)
    # print(cc.max(), cc.min())
    # import pdb;pdb.set_trace()
    # print(f"calculate r for {os.path.basename(reference_file)}")

    if swin:
        interested_region = (1,) if biomarker == 'ntr' else (1, 3) # JJ: for swin_unetr
        # print("for SWINUNETR annotation: nec is 1")
    else:
        interested_region = (2,) if biomarker == 'ntr' else (2, 3)

    "analyze mean/var"
    # gt
    wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
    nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, interested_region)  # JJ, ntr, ctr

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
    # calib-error（epsilon_y, epsilon_x
    r_naive, ce_left, ce_right, beta, opt_Q, opt_alpha = analyze_r_ce_std_alpha(
        tensor_nec_prob_map,
        tensor_wt_prob_map,
        tensor_nec_gt_map,
        tensor_wt_gt_map,
        tensor_seg_prob_map,
        tensor_seg_gt_map,
        ce_type,
        epsilon_y_mean_ece,  # for bound
        epsilon_x_mean_ece,
        total_out_prob)

    r_1_ord_corr, r_2_ord_corr = 0, 0
    # epsilon_y, epsilon_x = 0, 0
    epsilon_y, epsilon_x = calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map,
                                         tensor_wt_gt_map, bins=15)
    # a prior is: r>0, which is appicable for confidence range
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results["wt_size"] = wt_gt_counter
    results['ratio'] = {'r_gt': {}, 'v_bias': {}, f'epsilon_ece_{ce_type}': {},
                        'r_naive': {}}
    # gt
    r_gt = nec_gt_counter / wt_gt_counter
    results['ratio']['r_gt']['r_gt'] = r_gt
    results['ratio']['r_gt']['beta'] = beta
    # V-Bias: y,x
    # v_bias
    results['ratio']['v_bias']['v_bias_y_L1'] = v_bias_y_L1
    results['ratio']['v_bias']['v_bias_x_L1'] = v_bias_x_L1
    # CE: y,x
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] = epsilon_y.item()
    results['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] = epsilon_x.item()

    bounds_naive = generate_confidence_bounds(r_naive, ce_left, ce_right, beta)
    update_r_keys(results, 'r_naive', r_naive, r_gt,opt_Q, *sum(bounds_naive, ()))
    return results


def compute_estimator_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                                image_reader_writer: BaseReaderWriter,
                                file_ending: str,
                                regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                ignore_label: int = None,
                                num_processes: int = default_num_processes,
                                chill: bool = True,
                                binary: bool = False,
                                biomarker: str = 'ntr',
                                ce_type: str = "bins",  # "kde", nll, bs
                                ece_percentage: int = None,
                                total_out_prob: float = None,
                                swin: bool = False,
                                ) -> dict:
    """
    output_file must end with .json; can be None
    """
    alpha = None
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
    if alpha is None:
        output_file = output_file.replace('.json',f'_{total_out_prob}.json')

    os.makedirs(folder_save, exist_ok=True)

    val_json = load_json(join(folder_save.replace('test', 'validation_ece'), f'bins15_ratio.json')) 

    if ce_type == "v_bias":
        ## V_bias ##
        epsilon_y_mean_ece = np.array(
            [abs(per_case["ratio"]["v_bias"]["v_bias_y_L1"]) for per_case in val_json["ratio_per_case"]])
        epsilon_x_mean_ece = np.array(
            [abs(per_case["ratio"]["v_bias"]["v_bias_x_L1"]) for per_case in val_json["ratio_per_case"]])
    else:
        ## CE ##
        epsilon_y_mean_ece = np.array([per_case["ratio"]["epsilon_ece_bins15"]["epsilon_y"] for per_case in val_json["ratio_per_case"]])
        epsilon_x_mean_ece = np.array([per_case["ratio"]["epsilon_ece_bins15"]["epsilon_x"] for per_case in val_json["ratio_per_case"]])


    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    ################################################
    i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, biomarker, ce_type, epsilon_y_mean_ece, epsilon_x_mean_ece, alpha=alpha,
                                   source_dict = None, total_out_prob=total_out_prob,swin=swin)
        # i += 1
        # if i > 10: break
        results.append(result)

    ################################################
    paired_samples = {'r_gt': {}, 'r_naive': {},'opt_Q':{},
                      f'epsilon_ece_{ce_type}': {}, 'v_bias': {}}
    # r_gt and ref.nii.gz
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]

    ## tumor_size_mask
    paired_samples["wt_size"] = np.array([item['wt_size'] for item in results])
    n_samples = len(paired_samples["wt_size"])
    s_mask, m_mask, l_mask = s_m_l(paired_samples["wt_size"])
    ## ECE_mask
    # paired_samples["ece_x"] = np.array([item['ratio'][f'epsilon_ece_bins15']['epsilon_x'] for item in results]) # ECE_ranking
    # s_mask, m_mask, l_mask = s_m_l(paired_samples["ece_x"])

    # s_threshold, m_threshold = np.quantile(paired_samples["wt_size"], 0.33), np.quantile(paired_samples["wt_size"], 0.66)
    # # print("S/M/L:", s_threshold, m_threshold, "...")
    # s_mask = paired_samples["wt_size"] <= s_threshold
    # m_mask = (paired_samples["wt_size"] > s_threshold) & (paired_samples["wt_size"] <= m_threshold)
    # l_mask = paired_samples["wt_size"] > m_threshold
    # n_samples = len(paired_samples["wt_size"])

    # s_threshold, m_threshold = np.quantile(paired_samples["ece_x"], 0.33), np.quantile(paired_samples["wt_size"], 0.66)
    # # print("S/M/L:", s_threshold, m_threshold, "...")
    # s_mask = paired_samples["wt_size"] <= s_threshold
    # m_mask = (paired_samples["wt_size"] > s_threshold) & (paired_samples["wt_size"] <= m_threshold)
    # l_mask = paired_samples["wt_size"] > m_threshold



    ## Range: as narrow as possible ##
    mean_r_range = {}
    # mean_r_bias = {}
    MAE, MSE = {}, {}
    mean_opt_Q = {}
    scores = ['r_naive']
    for r in scores:
        paired_samples[r]['bound__ce'] = np.array([item['ratio'][r]['bound__ce'] for item in results])  # [N,2]
        paired_samples[r]['bound__ce+1std'] = np.array(
            [item['ratio'][r]['bound__ce+1std'] for item in results])  # [N,2]
        paired_samples[r]['range__ce+1std'] = np.array([item['ratio'][r]['range__ce+1std'] for item in results])  # [N,]
        paired_samples[r]['AE_r'] = np.array([item['ratio'][r]['AE_r'] for item in results])
        paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
        paired_samples[r]['opt_Q'] = np.array([item['ratio'][r]['opt_Q'] for item in results])  # opt_Q
        mean_opt_Q[r] = np.mean(paired_samples[r]['opt_Q'])
        mean_r_range[r] = np.mean(paired_samples[r]['range__ce+1std'], axis=0)
        # mean_r_bias[r] = np.mean(paired_samples[r]['bias_r'])
        MAE[r] = np.mean(paired_samples[r]['AE_r'])
        tumor_ece = np.array(
            [item["ratio"]["epsilon_ece_bins15"]["epsilon_x"] for item in results])

        
    ## tumor_size_mask
    mean_r_range['s'] = np.mean(paired_samples[r]['range__ce+1std'][s_mask], axis=0)
    mean_r_range['m'] = np.mean(paired_samples[r]['range__ce+1std'][m_mask], axis=0)
    mean_r_range['l'] = np.mean(paired_samples[r]['range__ce+1std'][l_mask], axis=0)

    MAE['s'] = np.mean(paired_samples[r]['AE_r'][s_mask])
    MAE['m'] = np.mean(paired_samples[r]['AE_r'][m_mask])
    MAE['l'] = np.mean(paired_samples[r]['AE_r'][l_mask])

    MSE['s'] = np.mean(paired_samples[r]['AE_r'][s_mask]**2)
    MSE['m'] = np.mean(paired_samples[r]['AE_r'][m_mask]**2)
    MSE['l'] = np.mean(paired_samples[r]['AE_r'][l_mask]**2)

    # cali-error for y and x
    mean_epsilon = {}
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_y'] for item in results])
    paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'] = np.array(
        [item['ratio'][f'epsilon_ece_{ce_type}']['epsilon_x'] for item in results])
    mean_epsilon['epsilon_y'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'])
    mean_epsilon['epsilon_x'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'])

    mean_epsilon['epsilon_y_s'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'][s_mask])
    mean_epsilon['epsilon_y_m'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'][m_mask])
    mean_epsilon['epsilon_y_l'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_y'][l_mask])
    mean_epsilon['epsilon_x_s'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'][s_mask])
    mean_epsilon['epsilon_x_m'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'][m_mask])
    mean_epsilon['epsilon_x_l'] = np.mean(paired_samples[f'epsilon_ece_{ce_type}']['epsilon_x'][l_mask])

    # bias for y and x
    mean_v_bias = {}
    paired_samples['v_bias']['v_bias_y_L1'] = np.array([item['ratio']['v_bias']['v_bias_y_L1'] for item in results])
    paired_samples['v_bias']['v_bias_x_L1'] = np.array([item['ratio']['v_bias']['v_bias_x_L1'] for item in results])
    mean_v_bias['v_bias_y_L1'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'])
    mean_v_bias['v_bias_x_L1'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'])

    mean_v_bias['v_bias_y_s'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'][s_mask])
    mean_v_bias['v_bias_x_s'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'][s_mask])
    mean_v_bias['v_bias_y_m'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'][m_mask])
    mean_v_bias['v_bias_x_m'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'][m_mask])
    mean_v_bias['v_bias_y_l'] = np.mean(paired_samples['v_bias']['v_bias_y_L1'][l_mask])
    mean_v_bias['v_bias_x_l'] = np.mean(paired_samples['v_bias']['v_bias_x_L1'][l_mask])


    # failure
    r_gt = paired_samples['r_gt']['r_gt']
    case_ids = np.array([
        os.path.basename(name).split('_')[-1].split('.')[0]
        for name in paired_samples['reference_file']
    ])
    bounds = paired_samples['r_naive']['bound__ce+1std']

    failure = {'all': detect_failure(paired_samples),
               's': detect_failure_size(r_gt[s_mask], case_ids[s_mask], bounds[s_mask]),
               'm': detect_failure_size(r_gt[m_mask], case_ids[m_mask], bounds[m_mask]),
               'l': detect_failure_size(r_gt[l_mask], case_ids[l_mask], bounds[l_mask]),
               }
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    head_results = [mean_r_range, MAE, MSE, mean_epsilon, mean_v_bias, failure]
    [recursive_fix_for_json_export(i) for i in head_results]

    result = {f'mean_ece_{ce_type}': mean_epsilon, 'mean_v_bias': mean_v_bias,
              'mean_opt_Q':mean_opt_Q,
              'MAE': MAE,
              'MSE': MSE,
              'mean_r_range__all': mean_r_range,
              'failure': failure,
              'ratio_per_case': results,
              }

    print('-----------------------------')
    # print('MAE_r: ', MAE)
    # print('MAE S/M/L: ', MAE['s'], MAE['m'], MAE['l'])
    print('-----------------------------')
    # print('MSE_r: ', MSE)
    print('MSE S/M/L: ', MSE['s'], MSE['m'], MSE['l'])
    # print('-----------------------------')
    print('v_bias_y S/M/L: ', mean_v_bias['v_bias_y_s'], mean_v_bias['v_bias_y_m'], mean_v_bias['v_bias_y_l'])
    print('v_bias_x S/M/L: ', mean_v_bias['v_bias_x_s'], mean_v_bias['v_bias_x_m'], mean_v_bias['v_bias_x_l'])
    # print('-----------------------------')
    # print(f'ECE: ', mean_epsilon['epsilon_y'], mean_epsilon['epsilon_x'])
    print('ECE_y S/M/L: ', mean_epsilon['epsilon_y_s'], mean_epsilon['epsilon_y_m'], mean_epsilon['epsilon_y_l'])
    print('ECE_x S/M/L: ', mean_epsilon['epsilon_x_s'], mean_epsilon['epsilon_x_m'], mean_epsilon['epsilon_x_l'])
    print('-----------------------------')
    # print('Range: ',round(mean_r_range['r_naive'],3))
    print('Range S/M/L: ', round(mean_r_range['s'],3), round(mean_r_range['m'],3), round(mean_r_range['l'],3))
    print('-----------------------------')
    # print("Cover: ", cover_rate(failure['all'], n_samples))
    print("Cover S/M/L: ", cover_rate(failure['s'], np.sum(s_mask)), cover_rate(failure['m'], np.sum(m_mask)), cover_rate(failure['l'], np.sum(l_mask)))

    ## ratio.json
    save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)
    # plot figures
    if 'fold_0' in folder_save:
        sigma = ''
        step_size = 10
        folder_save_hist = join(folder_save,"hist")
        os.makedirs(folder_save_hist, exist_ok=True)
        # plot_bins_dataset(paired_samples, folder_save, sigma, ce_type)  # "hist_of bias_and_range": bias_r, range
        plot_histogram_range(paired_samples['r_naive'][f'range__ce+1std'], title=f'Overall Confidence Interval (±sigma)',  # Interval
                             xlabel=f'Interval Widthth', save_path=join(folder_save_hist, f"{ce_type}_{int(total_out_prob*100)}.png"), color='#ba68c8')
        plot_ratio_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, int(total_out_prob*100))





    return paired_samples[f'epsilon_ece_{ce_type}']  # ['epsilon_ece_kde']['epsilon_y'] and 'x' for 2 np.arrays


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               binary: bool = False,
                               biomarker: str = 'ntr',
                               ce_type: str = "bins",  # "kde", nll, bs
                               ece_percentage: int = None,
                               total_out_prob: float = None,
                               swin: bool = False,
                               ):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']  # .nii.gz for segs
    # file_ending = ".npz" #.npz for probs

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = f'ratio_min_width.json'  # for min_width
        # output_file = f'ratio_min_width_ECEx_ranking.json'  # for min_width
        # output_file = f'1q_ratio_min_width.json'  # JJ

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                num_processes, chill=chill, binary=binary, biomarker=biomarker, ce_type=ce_type,
                                ece_percentage=ece_percentage, total_out_prob=total_out_prob,swin=swin) # 0.68


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
    parser.add_argument('--other_cal', type=str, required=False, default=None, help='IR, Direchlet')
    parser.add_argument('--ce_type', required=True, type=str, help='bins15, kde1, nll, bs')
    parser.add_argument('--biomarker', required=False, type=str, default="ntr", help='ntr, ctr')
    parser.add_argument('--ece_percentage', required=False, default=None, type=int,
                        help='e.g. 95 percentile of ECE_val')
    parser.add_argument('--total_out_prob', required=False, default=0.68, type=float)
    parser.add_argument('--swin', action='store_true', help='swin_unetr')
    args = parser.parse_args()
    if args.pfile is None:
        args.pfile = Path(args.pred_folder.rstrip("/")).parents[1] / "plans.json"
    if args.djfile is None:
        args.djfile = Path(args.pred_folder.rstrip("/")).parents[1] / "dataset.json"

    basename = os.path.basename(args.pred_folder)
    if args.TS is not None:
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_TS_{args.TS}')  # _TS_list_1000_new
    elif args.other_cal is not None:
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_{args.other_cal}')  # _IR
    # print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, binary=args.binary, biomarker=args.biomarker, ce_type=args.ce_type,
                               ece_percentage=args.ece_percentage, total_out_prob = args.total_out_prob, swin=args.swin)


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

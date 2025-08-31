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
from nnunetv2.evaluation.ece_utils import calc_ece_kde, calc_bs, calc_v_bias, calc_nll, calc_ece_bins, detect_failure, \
    ece_loss
from nnunetv2.evaluation.plot_utils import plot_ce_and_range_dataset, plot_ratio_and_range_dataset, plot_histogram_range
import torch.nn.functional as F
from nnunetv2.evaluation.seg_utils import gather_files, region_or_label_to_mask, region_or_label_to_mask_prob_add, \
    compute_tp_fp_fn_tn, get_ce_bound, convert_labels_to_one_hot, cat_all_source_files

def analyze_bound(tensor_nec_prob_map, tensor_wt_prob_map, total_out_prob, sub):
    tensor_nec_prob_map = tensor_nec_prob_map.reshape(-1)
    tensor_wt_prob_map = tensor_wt_prob_map.reshape(-1)
    n_voxels = tensor_nec_prob_map.size
    replace = False if sub<1.0 else True
    # r_est
    y_bar, x_bar  = np.mean(tensor_nec_prob_map), np.mean(tensor_wt_prob_map)
    r_est = y_bar/x_bar
    #####################
    # Bootstrapping
    ratios = []
    for _ in range(100):
        nec = np.random.choice(tensor_nec_prob_map, size=int(n_voxels*sub), replace=replace)
        wt = np.random.choice(tensor_wt_prob_map, size=int(n_voxels*sub), replace=replace)
        ntr = nec.mean()/wt.mean()
        ratios.append(ntr)
    ratios = np.array(ratios)
    # width = np.quantile(ratios, total_out_prob) # -> directly take quantile
    margin_prob = (1-total_out_prob)/2
    left, right = np.quantile(ratios, [margin_prob, 1-margin_prob])
    #####################
    return r_est, np.clip(left, 0, 1), np.clip(right, 0, 1)

def analyze_bound_box_region(tensor_nec_prob_map, tensor_wt_prob_map, total_out_prob):
    h, w, z = tensor_wt_prob_map.shape
    frac = 0.1
    n_bbox = int((1/frac)**3)
    # r_est
    y_bar, x_bar  = np.mean(tensor_nec_prob_map), np.mean(tensor_wt_prob_map)
    r_est = y_bar/x_bar
    #####################
    # Bootstrapping
    bh, bw, bz = int(frac * h), int(frac * w), int(frac * z)
    ratios = []
    for _ in range(100):
        nec, wt = [], []
        for _ in range(n_bbox):
            # random start_point
            h_start = np.random.randint(0, h - bh + 1)
            w_start = np.random.randint(0, w - bw + 1)
            z_start = np.random.randint(0, z - bz + 1)
            # end_point
            h_end = h_start + bh
            w_end = w_start + bw
            z_end = z_start + bz
            block_nec = tensor_nec_prob_map[h_start:h_end, w_start:w_end, z_start:z_end]
            block_wt = tensor_wt_prob_map[h_start:h_end, w_start:w_end, z_start:z_end]
            nec.append(block_nec.reshape(-1))
            wt.append(block_wt.reshape(-1))
        ntr = np.concatenate(nec, axis=0).mean()/np.concatenate(wt, axis=0).mean()
        ratios.append(ntr)
    ratios = np.array(ratios)
    margin_prob = (1-total_out_prob)/2
    left, right = np.quantile(ratios, [margin_prob, 1-margin_prob])
    #####################
    return r_est, np.clip(left, 0, 1), np.clip(right, 0, 1)

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



def clip_to_unit_range(x, y):  # range [0, 1]
    return np.clip(x, 0, 1), np.clip(y, 0, 1)


def generate_confidence_bounds(r, ce_left, ce_right, beta):
    """
    Generate CE, CE+1σ, CE+2σ, CE+3σ intervals.
    Returns: (x_ce, y_ce), (x1σ, y1σ), (x2σ, y2σ), (x3σ, y3σ)
    """
    x_ce, y_ce = clip_to_unit_range(r - ce_left, r + ce_right)
    x_1std, y_1std = clip_to_unit_range(x_ce - beta, y_ce + beta)
    return (x_ce, y_ce), (x_1std, y_1std)



def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      binary: bool = False,
                      biomarker: str = 'ntr',
                      ce_type: str = "bootstrap",
                      total_out_prob: float = None,
                      swin: bool = False,
                      sub: float = 1.0,
                      regional: bool = False,
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
        interested_region = (1,) if biomarker == 'ntr' else (1, 3)  # JJ: for swin_unetr
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

    tensor_nec_prob_map = nec_prob_map  # [155,240,240]
    tensor_wt_prob_map = wt_prob_map
    tensor_nec_gt_map = nec_gt_map.squeeze(0)
    tensor_wt_gt_map = wt_gt_map.squeeze(0)
    tensor_seg_prob_map = prob_pred
    tensor_seg_gt_map = seg_ref

    "reformulate r to see how interval(x,y) changes"
    # bound
    r_gt = nec_gt_counter / wt_gt_counter

    if regional:
        r_est, left, right = analyze_bound_box_region(tensor_nec_prob_map,tensor_wt_prob_map,total_out_prob)
    else:
        r_est, left, right = analyze_bound(tensor_nec_prob_map,tensor_wt_prob_map,total_out_prob, sub)
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results["wt_size"] = wt_gt_counter
    results['ratio'] = {'r_gt': {'r_gt': r_gt}, 'r_naive': {}}
    # bound
    results['ratio']['r_naive'] = {
        'r_est': r_est,
        'AE_r': abs(r_est - r_gt),
        'bound__ce+1std': np.array([left, right]),
        'range__ce+1std': right - left,
    }
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
                                ce_type: str = "bootstrap",  
                                ece_percentage: int = None,
                                total_out_prob: float = None,
                                swin: bool = False,
                                sub: float = 1.0,
                                regional: bool = False,
                                ) -> dict:

    folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
    output_file = output_file.replace('.json', f'_{total_out_prob}.json')

    os.makedirs(folder_save, exist_ok=True)
    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    ################################################
    i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label,
                                   binary, biomarker, ce_type, total_out_prob=total_out_prob, swin=swin, sub=sub, regional= regional)
        # i += 1
        # if i > 10: break
        results.append(result)

    ################################################
    paired_samples = {'r_gt': {}, 'r_naive': {}}
    # r_gt and ref.nii.gz
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    paired_samples["wt_size"] = np.array([item['wt_size'] for item in results])
    n_samples = len(paired_samples["wt_size"])
    ## tumor_size_mask
    s_threshold, m_threshold = np.quantile(paired_samples["wt_size"], 0.33), np.quantile(paired_samples["wt_size"],
                                                                                         0.66)
    # print("S/M/L:", s_threshold, m_threshold, "...")
    s_mask = paired_samples["wt_size"] <= s_threshold
    m_mask = (paired_samples["wt_size"] > s_threshold) & (paired_samples["wt_size"] <= m_threshold)
    l_mask = paired_samples["wt_size"] > m_threshold

    ## Range: as narrow as possible ##
    mean_r_range = {}
    MAE = {}
    mean_opt_Q = {}
    r = 'r_naive'
    paired_samples[r]['bound__ce+1std'] = np.array([item['ratio'][r]['bound__ce+1std'] for item in results])  # [N,2]
    paired_samples[r]['range__ce+1std'] = np.array([item['ratio'][r]['range__ce+1std'] for item in results])  # [N,]
    paired_samples[r]['AE_r'] = np.array([item['ratio'][r]['AE_r'] for item in results])
    paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
    mean_r_range[r] = np.mean(paired_samples[r]['range__ce+1std'], axis=0)
    MAE[r] = np.mean(paired_samples[r]['AE_r'])

    ## tumor_size_mask
    mean_r_range['s'] = np.mean(paired_samples[r]['range__ce+1std'][s_mask], axis=0)
    mean_r_range['m'] = np.mean(paired_samples[r]['range__ce+1std'][m_mask], axis=0)
    mean_r_range['l'] = np.mean(paired_samples[r]['range__ce+1std'][l_mask], axis=0)

    MAE['s'] = np.mean(paired_samples[r]['AE_r'][s_mask])
    MAE['m'] = np.mean(paired_samples[r]['AE_r'][m_mask])
    MAE['l'] = np.mean(paired_samples[r]['AE_r'][l_mask])

    # failure
    failure_1std = detect_failure(paired_samples)  # fail_case: outside the range
    failure = {}
    failure['std'] = failure_1std.tolist()
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    [recursive_fix_for_json_export(i) for i in [mean_r_range, MAE, failure]]

    result = {'MAE': MAE,
              'mean_r_range__all': mean_r_range,
              'failure': failure,
              'ratio_per_case': results,
              }

    # print('-----------------------------')
    # print('MAE_r: ', MAE)
    # print('MAE S/M/L: ', MAE['s'], MAE['m'], MAE['l'])
    print('-----------------------------')
    # print('Range: ',mean_r_range['r_naive'])
    print('S/M/L: ', round(mean_r_range['s'],3), round(mean_r_range['m'],3), round(mean_r_range['l'],3))
    print('-----------------------------')
    print("failure: ", len(failure['std']), "Cover.: ", round((1-len(failure['std'])/n_samples)*100, 3))

    ## ratio.json
    save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)
    # plot figures
    # if 'fold_0' in folder_save:
    #     sigma = ''
    #     step_size = 10
    #     folder_save_hist = join(folder_save, "hist")
    #     os.makedirs(folder_save_hist, exist_ok=True)
    #     # plot_bins_dataset(paired_samples, folder_save, sigma, ce_type)  # "hist_of bias_and_range": bias_r, range
    #     plot_histogram_range(paired_samples['r_naive'][f'range__ce+1std'],
    #                          title=f'Overall Confidence Interval (±sigma)',  # Interval
    #                          xlabel=f'Interval Widthth',
    #                          save_path=join(folder_save_hist, f"{ce_type}_{int(total_out_prob * 100)}.png"),
    #                          color='#ba68c8')
    #     plot_ratio_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, int(total_out_prob * 100))



def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               binary: bool = False,
                               biomarker: str = 'ntr',
                               ce_type: str = "bootstrap",  # "kde", nll, bs
                               ece_percentage: int = None,
                               total_out_prob: float = None,
                               swin: bool = False,
                               sub: float = 1.0,
                               regional: bool = False,
                               ):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']  # .nii.gz for segs

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        if regional:
            output_file = f'ratio_region.json'
        output_file = f'ratio_{sub}sub.json' if sub < 1.0 else 'ratio.json'

        lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                num_processes, chill=chill, binary=binary, biomarker=biomarker, ce_type=ce_type,
                                ece_percentage=ece_percentage, total_out_prob=total_out_prob, swin=swin, sub=sub, regional=regional)  # 0.68


if __name__ == '__main__':
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
    parser.add_argument('--ce_type', required=False, type=str, default="bootstrap", help='bins15, kde1, nll, bs')
    parser.add_argument('--biomarker', required=False, type=str, default="ntr", help='ntr, ctr')
    parser.add_argument('--ece_percentage', required=False, default=None, type=int,
                        help='e.g. 95 percentile of ECE_val')
    parser.add_argument('--total_out_prob', required=False, default=0.68, type=float)
    parser.add_argument('--sub', required=False, default=1.0, type=float)
    parser.add_argument('--swin', action='store_true', help='swin_unetr')
    parser.add_argument('--regional', action='store_true', help='regional_sampling')
    args = parser.parse_args()
    if args.pfile is None:
        args.pfile = Path(args.pred_folder.rstrip("/")).parents[1] / "plans.json"
    if args.djfile is None:
        args.djfile = Path(args.pred_folder.rstrip("/")).parents[1] / "dataset.json"

    basename = os.path.basename(args.pred_folder)
    if args.TS is not None:
        args.pred_folder = args.pred_folder.replace(basename, basename + f'_TS_{args.TS}')  # _TS_list_1000_new
    # print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, binary=args.binary, biomarker=args.biomarker, ce_type=args.ce_type,
                               ece_percentage=args.ece_percentage, total_out_prob=args.total_out_prob, swin=args.swin, sub=args.sub, regional =args.regional,)

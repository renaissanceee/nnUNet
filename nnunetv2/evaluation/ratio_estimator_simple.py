import multiprocessing
import os
from pathlib import Path
import re
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import ast
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
from nnunetv2.evaluation.ece_utils import detect_failure_std
from nnunetv2.evaluation.plot_utils import plot_corr_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset, \
    plot_ratio_and_range_dataset, plot_ratio_and_range_all
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from nnunetv2.evaluation.seg_utils import gather_files_seed, region_or_label_to_mask, region_or_label_to_mask_prob_add, \
    compute_tp_fp_fn_tn


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


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      biomarker: str = 'ntr',
                      seeds: list = [6664, 4435, 5606, 2944, 4734],
                      ) -> dict:
    r_est = []
    for seed in seeds:
        # str operation
        reference_file = reference_file
        # if 'dropout0.5' in prediction_file:
        #     prediction_file_cur = prediction_file.replace('nnUNetTrainerCELoss__nnUNetPlans__2d__dropout0.5',
        #                                                   f'nnUNetTrainerCELoss__nnUNetPlans__2d__dropout0.5__{seed}')
        #     probability_file_cur = probability_file.replace('nnUNetTrainerCELoss__nnUNetPlans__2d__dropout0.5',
        #                                                     f'nnUNetTrainerCELoss__nnUNetPlans__2d__dropout0.5__{seed}')
        # else:
        #     prediction_file_cur = prediction_file.replace('nnUNetTrainerCELoss__nnUNetPlans__2d',
        #                                                   f'nnUNetTrainerCELoss__nnUNetPlans__2d__{seed}')
        #     probability_file_cur = probability_file.replace('nnUNetTrainerCELoss__nnUNetPlans__2d',
        #                                                     f'nnUNetTrainerCELoss__nnUNetPlans__2d__{seed}')

        prediction_file_cur = prediction_file.replace('/fold_0',f'__{seed}/fold_0')
        probability_file_cur = probability_file.replace('/fold_0',f'__{seed}/fold_0')

        # load images
        seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
        seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file_cur)
        print(probability_file_cur)
        prob_pred = np.load(probability_file_cur)['probabilities']  # (3,155,240,240) within [0,1]
        ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

        # print(f"calculate r for {os.path.basename(reference_file)}")
        interested_region = (2,) if biomarker == 'ntr' else (2, 3)

        "analyze mean/var"
        # gt
        wt_gt_map, wt_gt_counter = region_or_label_to_mask(seg_ref, (1, 2, 3))
        nec_gt_map, nec_gt_counter = region_or_label_to_mask(seg_ref, interested_region)  # JJ, ntr, ctr

        # pred
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
        # sigma
        y_bar, x_bar, cov_x_y, cov_x2_y, cov_y2_x, cov_x2_x, var_x, var_y, n = calc_statistic(tensor_nec_prob_map,
                                                                                              tensor_wt_prob_map)

        r_est.append((y_bar / x_bar).item())

    # a prior is: r>0, which is appicable for confidence range
    #####################################
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'r_gt': {}, 'r_naive': {}}

    r_gt = nec_gt_counter / wt_gt_counter
    mean, std = np.mean(r_est), np.std(r_est)
    bounds_std = np.clip([mean - std, mean + std], 0, 1)
    bounds_3std = np.clip([mean - 3*std, mean + 3*std], 0, 1)
    results['ratio']['r_gt']['r_gt'] = r_gt
    results['ratio']['r_naive']['r_est_all'] = r_est
    results['ratio']['r_naive']['r_est'] = mean
    results['ratio']['r_naive']['bound__std'] = bounds_std
    results['ratio']['r_naive']['range__std'] = bounds_std[1] - bounds_std[0]
    results['ratio']['r_naive']['bound__3std'] = bounds_3std
    results['ratio']['r_naive']['range__3std'] = bounds_3std[1] - bounds_3std[0]
    return results


def compute_estimator_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                                image_reader_writer: BaseReaderWriter,
                                file_ending: str,
                                regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                ignore_label: int = None,
                                num_processes: int = default_num_processes,
                                chill: bool = True,
                                biomarker: str = 'ntr',
                                seeds: list = [],
                                ) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")


    # folder_save = folder_save.replace('nnUNetTrainerCELoss__nnUNetPlans__2d',
    #                                   'nnUNetTrainerCELoss__nnUNetPlans__2d__ensemble')
    print(f'save into {folder_save}')
    os.makedirs(folder_save, exist_ok=True)

    files_pred, files_prob, files_ref = gather_files_seed(folder_pred, folder_ref, ".nii.gz")
    results = []
    ################################################
    i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label, biomarker,
                                   seeds)
        # i += 1
        # if i > 2: break
        results.append(result)

    ################################################
    paired_samples = {'r_gt': {}, 'r_naive': {}}
    # r_gt and ref.nii.gz
    paired_samples['r_gt']['r_gt'] = np.array([item['ratio']['r_gt']['r_gt'] for item in results])  # [N,]
    paired_samples['reference_file'] = np.array([item['reference_file'] for item in results])  # [N,]
    ## Range: as narrow as possible ##
    r = 'r_naive'
    paired_samples[r]['bound__std'] = np.array([item['ratio'][r]['bound__std'] for item in results])  # [N,2]
    paired_samples[r]['range__std'] = np.array([item['ratio'][r]['range__std'] for item in results])  # [N,]
    paired_samples[r]['bound__3std'] = np.array([item['ratio'][r]['bound__3std'] for item in results])  # [N,2]
    paired_samples[r]['range__3std'] = np.array([item['ratio'][r]['range__3std'] for item in results])  # [N,]
    paired_samples[r]['r_est'] = np.array([item['ratio'][r]['r_est'] for item in results])  # [N,]
    mean_r_range = {'std': np.mean(paired_samples[r]['range__std'], axis=0), '3std': np.mean(paired_samples[r]['range__3std'], axis=0)}

    # failure
    failure_std, failure_3std = detect_failure_std(paired_samples)  # fail_case: outside the range
    failure = {'std': failure_std.tolist(), '3std': failure_3std.tolist()}
    # sort for json
    [recursive_fix_for_json_export(i) for i in results]
    [recursive_fix_for_json_export(i) for i in [mean_r_range, failure]]

    result = {'mean_r_range': mean_r_range,
              'failure': failure,
              'ratio_per_case': results,
              }

    print('-----------------------------')
    print('range: ', mean_r_range['std'], mean_r_range['3std'])
    print('-----------------------------')
    print("failure:", len(failure['std']), len(failure['3std']))
    ## ratio.json
    save_json(result, join(folder_save, f"std_{output_file}"), sort_keys=False)
    # plot figures
    # if 'fold_0' in folder_save:
    #     sigma = ''
    #     step_size = 10
    #     plot_bins_dataset(paired_samples, folder_save, sigma, ce_type,
    #                       ece_percentage)  # "hist_of bias_and_range": bias_r, range
    #     plot_ratio_and_range_dataset(paired_samples, folder_save, sigma, step_size, ce_type, ece_percentage)


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False,
                               biomarker: str = 'ntr',
                               seeds: list = [],
                               ):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']  # .nii.gz for segs
    # file_ending = ".npz" #.npz for probs

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = f'ratio.json'  # 5 times ensemble

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_estimator_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                                num_processes, chill=chill, biomarker=biomarker, seeds=seeds)


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
    parser.add_argument('--TS', type=str, required=False, default=None, help='Temperature Scaling')
    parser.add_argument('--other_cal', type=str, required=False, default=None, help='IR, Direchlet')
    parser.add_argument('--biomarker', required=False, type=str, default='ntr', help='ntr, ctr')
    parser.add_argument('--seeds', type=str, required=False, default='[1,2,3]')

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
                               chill=args.chill, biomarker=args.biomarker, seeds=ast.literal_eval(args.seed))


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
    parser.add_argument('--biomarker', required=False, type=str, default='ntr', help='ntr, ctr')
    parser.add_argument('--seeds', type=str, required=False, default='[1,2,3]')

    args = parser.parse_args()
    # args.gt_folder = '/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/'
    # args.pred_folder = f'/staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test'
    args.pfile = Path(args.pred_folder.rstrip("/")).parents[1] / "plans.json"
    args.djfile = Path(args.pred_folder.rstrip("/")).parents[1] / "dataset.json"
    basename = os.path.basename(args.pred_folder)
    # print(f'calculating from ... {args.pred_folder}')
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np,
                               chill=args.chill, biomarker=args.biomarker, seeds=ast.literal_eval(args.seeds))

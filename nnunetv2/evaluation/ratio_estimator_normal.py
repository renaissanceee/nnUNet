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
# from nnunetv2.evaluation.ece_utils import calc_ece_kde,calc_bs,calc_v_bias,calc_nll,calc_ece_bins
from nnunetv2.evaluation.ece_utils import get_ece_bins,fast_ece,ece_loss, l1_score,get_ece_kde_sub,brier_score
# from nnunetv2.evaluation.plot_utils import plot_r_and_range_dataset, plot_ce_and_range_dataset, plot_bins_dataset
from sklearn.metrics import accuracy_score, log_loss
import torch.nn.functional as F
from nnunetv2.evaluation.seg_utils import gather_files, region_or_label_to_mask, region_or_label_to_mask_prob_add,\
    compute_tp_fp_fn_tn,get_ce_bound


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


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str,
                      image_reader_writer: BaseReaderWriter,
                      labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                      ignore_label: int = None,
                      ce_type: str = "bins",
                      ) -> dict:
    print(f'calc for {os.path.basename(probability_file)}')
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)  # (1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    tensor_seg_prob_map = torch.from_numpy(prob_pred)
    tensor_seg_gt_map = torch.from_numpy(seg_ref)

    # calib-error
    tensor_seg_prob_map, tensor_seg_gt_map = tensor_seg_prob_map.permute(1, 2, 3, 0).reshape(-1, 4), tensor_seg_gt_map.reshape(-1)
    tensor_seg_gt_map_onehot = F.one_hot(tensor_seg_gt_map.long(), num_classes=4)

    list_kde=[]

    for i in range(5):
        epsilon_kde = get_ece_kde_sub(tensor_seg_prob_map, tensor_seg_gt_map.to(torch.int64),bandwidth=0.02, p=1,mc_type='top_label', device='cuda', sub=1e4) # 'kde'
        list_kde.append(epsilon_kde.item())
    epsilon_kde = np.mean(list_kde, axis=0)
    # 'bins'
    confidences, predictions = torch.max(tensor_seg_prob_map, dim=1);accuracies = predictions.eq(tensor_seg_gt_map)
    epsilon_bin = get_ece_bins(confidences[:,None], accuracies,  bins=15, device="cuda")
    # 'bins_loss'
    epsilon_bin_loss = ece_loss(tensor_seg_prob_map, tensor_seg_gt_map, bins=15)
    # 'bs': one-hot
    epsilon_bs = brier_score(tensor_seg_prob_map, tensor_seg_gt_map_onehot)
    ## 'nll'
    #epsilon_nll = calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map)
    #####################################

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {'ece_kde': epsilon_kde,'ece_bin': epsilon_bin.item(),'ece_bin_loss': epsilon_bin_loss.item(),'bs': epsilon_bs.item(), }
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

    folder_save = join(folder_pred, f"ratio_metrics_prob_{biomarker}")
    os.makedirs(folder_save, exist_ok=True)
    files_pred, files_prob, files_ref = gather_files(folder_pred, folder_ref, ".nii.gz", chill)
    results = []
    ################################################
    i = 0
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        # i += 1
        # if i > 10 or i < 8: continue  # JJ: first two samples
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label, ce_type)
        results.append(result)

    ################################################
    # results['ratio'] = {'ece_kde': epsilon_kde.item(), 'ece_bin': epsilon_bin.item(), 'bs': epsilon_bs.item(), }
    mean_kde= np.mean(np.array([item['ratio']['ece_kde']for item in results]))
    mean_bin = np.mean(np.array([item['ratio']['ece_bin'] for item in results]))
    mean_bin_loss = np.mean(np.array([item['ratio']['ece_bin_loss'] for item in results]))
    mean_bs = np.mean(np.array([item['ratio']['bs'] for item in results]))
    mean_epsilon = {'mean_ece_kde': mean_kde, 'mean_ece_bin': mean_bin, 'mean_ece_bin_loss': mean_bin_loss, 'mean_bs': mean_bs}

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(mean_epsilon)


    result = {f'mean_ece': mean_epsilon, 'ratio_per_case': results}
    save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)

    # [recursive_fix_for_json_export(i) for i in results]
    # head_results = [mean_epsilon_kde, mean_epsilon_bin, mean_epsilon_bin]
    # [recursive_fix_for_json_export(i) for i in head_results]
    # result = {'mean_ece_kde': mean_epsilon_kde, 'ece_bin': mean_epsilon_bin, 'bs': mean_bs }
    # save_json(result, join(folder_save, f"{ce_type}_{output_file}"), sort_keys=False)

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
        # output_file = 'cali_error.json'
        output_file = 'bins.json'

    lm = PlansManager(plans_file).get_label_manager(dataset_json)

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
    parser.add_argument('--ce_type', required=False, type=str, help='bins15, kde1, nll, bs')
    parser.add_argument('--biomarker', required=False, type=str, help='ntr, ctr')
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

import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

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

#  from: https://github.com/AxelJanRousseau/PostTrainCalibration/
def fast_ece(y_true, y_pred, n_bins=10):
    # ~sklearn code
    bins = np.linspace(0., 1. - 1./n_bins, n_bins)
    binids = np.digitize(y_pred, bins) - 1

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0  # don't use empty bins
    prob_true = (bin_true[nonzero] / bin_total[nonzero])  # acc
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])  # conf
    weights = bin_total[nonzero] / np.sum(bin_total[nonzero])
    l1 = np.abs(prob_true-prob_pred)
    ece = np.sum(weights*l1)
    mce = l1.max()
    l1 = l1.sum()
    return {"acc": prob_true, "conf": prob_pred, "ECE": ece, "MCE": mce, "l1": l1}

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
    # convert keys in bias_and_ece metrics
    results_converted['bias_and_ece'] = {label_or_region_to_key(k): results['bias_and_ece'][k] for k in results['bias_and_ece'].keys()}
    # convert ratio_per_case
    for i in range(len(results_converted["ratio_per_case"])):
        results_converted["ratio_per_case"][i]['ratio'] = \
            {label_or_region_to_key(k): results["ratio_per_case"][i]['ratio'][k]
             for k in results["ratio_per_case"][i]['ratio'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=False)#  sort_keys=True


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in bias_and_ece metrics
    results['bias_and_ece'] = {key_to_lgabel_or_region(k): results['bias_and_ece'][k] for k in results['bias_and_ece'].keys()}
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
        for r in region_or_label:# 1,2,3
            mask[segmentation == r] = True
    return mask, np.count_nonzero(mask)

def region_or_label_to_mask_prob_max(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(mask_ref)
        for r in region_or_label:# 1,2,3
            cur_seg_channel_map = segmentation[r-1,:,:,:][None,...]
            mask = np.maximum(mask, cur_seg_channel_map) # take max confidence over 3-channels
    mask[~mask_ref] = 0 # only consider tp pixels
    return mask, np.sum(mask)

def region_or_label_to_mask_prob_multiply(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros(segmentation.shape[1:])
        for r in region_or_label:  # 1,2,3
            if r == region_or_label[0]:
                mask = segmentation[r-1, :, :, :][None, ...]  # init as non-zero
            else:
                cur_seg_channel_map = segmentation[r-1, :, :, :][None, ...]
                mask = mask * cur_seg_channel_map  # take prduction over 3-channels
    # print(f'Denominator. Production is {np.sum(mask)}')
    return mask, np.sum(mask)

def region_or_label_to_mask_prob_add(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros(segmentation.shape[1:])
        for r in region_or_label:  # 1,2,3
            mask = mask + segmentation[r, :, :, :]# mask[z,x,y], seg:4-channel
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

def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    # print(np.count_nonzero(seg_ref == 0.0))
    # print(np.count_nonzero(seg_ref == 1.0))
    # print(np.count_nonzero(seg_ref == 2.0))
    # print(np.count_nonzero(seg_ref == 3.0))
    ## sum up to be 240*240*155

    for r in labels_or_regions:#  [(1,2,3), (2,3), (3,)]
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)# np_array(1,155,240,240)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def compute_estimator(reference_file: str, prediction_file: str, probability_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file) #(1,155,240,240) within {0.0,1.0,2.0,3.0}
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    prob_pred = np.load(probability_file)['probabilities']  # (3,155,240,240) within [0,1]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['probability_file'] = probability_file
    results['ratio'] = {}
    wt_prob_counter,wt_counter = 0, 0
    # JJ
    labels_or_regions=[(1,2,3), (2,3), (2,)] # add (2,) as necrosis
    for r in labels_or_regions: # [(1,2,3), (2,3), (3,)]
        # per-sample: prob-ratio
        prob_map, prob_counter = region_or_label_to_mask_prob_add(prob_pred, r)  # joint_prob
        ## mean? no, just sum up
        class_map, class_counter = region_or_label_to_mask(seg_ref, r) # joint_gt
        if r == (1, 2, 3):# denominator: 1∪2∪3
            wt_prob_counter = prob_counter
            wt_counter = class_counter
        if r == (2, 3):# 2∪3
            results['ratio']['pred_CTR_naive'] = prob_counter/wt_prob_counter# ['naive_core_wt_ratio']
            results['ratio']['gt_CTR'] = class_counter / wt_counter# ['gt_core_wt_ratio']
        if r == (2,):# 2
            results['ratio']['pred_NTR_naive'] = prob_counter/wt_prob_counter # ['naive_necrosis_wt_ratio']
            results['ratio']['gt_NTR'] = class_counter / wt_counter# ['gt_necrosis_wt_ratio']

    return results


def compute_estimator_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
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
    files_prob = [join(folder_pred, i.replace("nii.gz","npz")) for i in files_pred]
    # with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
    #     # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
    #     #     compute_metrics(*i)
    #     results = pool.starmap(
    #         compute_metrics,
    #         list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
    #                  [ignore_label] * len(files_pred)))
    #     )

    results = []
    for ref, pred, prob in zip(files_ref, files_pred, files_prob):
        # result = compute_metrics(ref, pred, image_reader_writer, regions_or_labels, ignore_label) # also do
        result = compute_estimator(ref, pred, prob, image_reader_writer, regions_or_labels, ignore_label)
        results.append(result)

    ## mean metric per class
    ## metric_list = list(results[0]['ratio'][regions_or_labels[0]].keys()) # original(region-based)
    # metric_list = list(results[0]['ratio'][tuple(regions_or_labels)].keys())  # foreground(label-based)# regions_or_labels = [1,2,3]
    ## [1,2,3]
    # means = {}
    # for r in regions_or_labels:
    #     means[r] = {}
    #     for m in metric_list:
    #         means[r][m] = np.nanmean([i['ratio'][r][m] for i in results])

    # means = {}
    # for r in ratios:
    #     means[r] = {}
    #     means[r] = np.nanmean([item['ratio'][r] for item in results])

    ###################
    ratios = ['pred_NTR_naive','pred_CTR_naive','gt_NTR','gt_CTR']
    paired_samples = {}
    for r in ratios:  # re-arrange
        paired_samples[r] = {}
        paired_samples[r] = np.array([item['ratio'][r] for item in results])
    ###################
    ## bias&ece over dataset ##
    # ratio-bias
    bias_and_ece = {}
    bias_and_ece['l1_bias_NTR'] = np.mean(paired_samples['pred_NTR_naive']-paired_samples['gt_NTR'])
    bias_and_ece['l1_bias_CTR'] = np.mean(paired_samples['pred_CTR_naive']-paired_samples['gt_CTR'])
    # ratio-ECE:
    bias_and_ece['ece_bins_NTR'] = fast_ece(paired_samples['gt_NTR'], paired_samples['pred_NTR_naive'], n_bins=20)["ECE"] # JJ: Mar.02
    bias_and_ece['ece_bins_CTR'] = fast_ece(paired_samples['gt_CTR'], paired_samples['pred_CTR_naive'], n_bins=20)["ECE"]

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(bias_and_ece)
    result = {'bias_and_ece': bias_and_ece,'ratio_per_case': results}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending'] #.nii.gz for segs
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
                              num_processes, chill=chill)


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
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)


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
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)

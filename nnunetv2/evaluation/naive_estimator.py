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

def plot_gaussian_distr(r_samples, mu_r, var_r, save_file_path):
    plt.hist(r_samples, bins=50, density=True, alpha=0.6, color='b', label="Empirical Distribution")
    plt.axvline(mu_r, color='r', linestyle='dashed', linewidth=2, label="Theoretical Mean")
    plt.axvline(mu_r - 3 * np.sqrt(var_r), color='orange', linestyle='dotted', linewidth=2,
                label="Theoretical 3σ range")
    plt.axvline(mu_r + 3 * np.sqrt(var_r), color='orange', linestyle='dotted', linewidth=2)
    plt.xlabel("r = x̄ / ȳ")
    plt.ylabel("Density")
    plt.title("Distribution of r")
    plt.xlim(0.12, 0.25)
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    results['ratio'] = {'naive_ratio':{},'second_corr_ratio':{}}
    # results['ratio']['naive_ratio']['gaussian_mu_y'] = 23


    "consider CE_r,bias_r"
    # wt_prob_counter,wt_counter = 0, 0
    # labels_or_regions = [(1,2,3), (2, 3), (2,)]
    # for r in labels_or_regions: # [(1,2,3), (2,3), (3,)]
    #     # per-sample: prob-ratio
    #     prob_map, prob_counter = region_or_label_to_mask_prob_add(prob_pred, r)  # joint_prob
    #     ## mean? no, just sum up
    #     class_map, class_counter = region_or_label_to_mask(seg_ref, r) # joint_gt
    #     if r == (1, 2, 3):# denominator: 1∪2∪3
    #         wt_prob_counter = prob_counter
    #         wt_counter = class_counter
    #     if r == (2, 3):# 2∪3
    #         results['ratio']['pred_CTR_naive'] = prob_counter/wt_prob_counter# ['naive_core_wt_ratio']
    #         results['ratio']['gt_CTR'] = class_counter / wt_counter# ['gt_core_wt_ratio']
    #     if r == (2,):# 2
    #         results['ratio']['pred_NTR_naive'] = prob_counter/wt_prob_counter # ['naive_necrosis_wt_ratio']
    #         results['ratio']['gt_NTR'] = class_counter / wt_counter# ['gt_necrosis_wt_ratio']
    # return results



    "no gt, just analyze mean/var"
    labels_or_regions = [(2, 3), (2,)]  # (2,3):core, (2,):necrosis
    wt_prob_map, wt_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (1, 2, 3))  # denominator
    nec_prob_map, nec_prob_counter = region_or_label_to_mask_prob_add(prob_pred, (2, ))  # numerator
    n = np.size(wt_prob_map)//100 # 8928000
    ### JJ: np.array
    # n=1000
    # normal_distr: mean+variance
    ## x,y
    # mu_y, var_y= np.mean(nec_prob_map),np.var(nec_prob_map)
    # mu_x, var_x= np.mean(wt_prob_map),np.var(wt_prob_map)
    # cov_xy = np.cov(wt_prob_map.flatten(), nec_prob_map.flatten())[0, 1]
    # ## r=y/x
    # mu_r = mu_y / mu_x * (1 + (1 / n) * (var_x / mu_x**2 - cov_xy / (mu_x * mu_y)))
    # var_r = (mu_y**2 / mu_x**4 * var_x  + var_y / mu_x**2 - 2 * mu_y / mu_x**3 * cov_xy )/ n
    # ## mc-simulation
    # n_samples = 10000
    # # import pdb;pdb.set_trace()
    # # plot
    # x_samples = np.random.normal(mu_x, np.sqrt(var_x), (n_samples, n))
    # y_samples = np.random.normal(mu_y, np.sqrt(var_y), (n_samples, n))
    # y_bar,x_bar = y_samples.mean(axis=1),x_samples.mean(axis=1)  # (10000,10000) -> (10000,)
    # r_samples = y_bar / x_bar
    # empirical_mean, empirical_std = np.mean(r_samples), np.std(r_samples)
    # # r-distr
    # plt.hist(r_samples, bins=50, density=True, alpha=0.6, color='b', label="Empirical Distribution")
    # plt.axvline(mu_r, color='r', linestyle='dashed', linewidth=2, label="Theoretical Mean")
    # # 3σ-range
    # plt.axvline(mu_r - 3 * np.sqrt(var_r), color='orange', linestyle='dotted', linewidth=2,label="Theoretical 3σ range")
    # plt.axvline(mu_r + 3 * np.sqrt(var_r), color='orange', linestyle='dotted', linewidth=2)
    # plt.xlabel("r = x̄ / ȳ")
    # plt.ylabel("Density")
    # plt.title("Distribution of r")
    # plt.xlim(0, 1)
    # plt.legend()
    #####################################
    ### JJ: torch.tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nec_prob_map = torch.from_numpy(nec_prob_map).float().to(device)
    wt_prob_map = torch.from_numpy(wt_prob_map).float().to(device)
    mu_y, var_y = torch.mean(nec_prob_map), torch.var(nec_prob_map)
    mu_x, var_x = torch.mean(wt_prob_map), torch.var(wt_prob_map)
    cov_x_y = torch.cov(torch.stack([wt_prob_map.flatten(), nec_prob_map.flatten()]))[0, 1]
    mu_r = mu_y / mu_x * (1 + (1 / n) * (var_x / mu_x ** 2 - cov_x_y / (mu_x * mu_y)))
    var_r = (mu_y ** 2 / mu_x ** 4 * var_x + var_y / mu_x ** 2 - 2 * mu_y / mu_x ** 3 * cov_x_y) / n
    # MC
    n_samples = 10000
    x_samples = torch.normal(mu_x, torch.sqrt(var_x), size=(n_samples, n), device=device)
    y_samples = torch.normal(mu_y, torch.sqrt(var_y), size=(n_samples, n), device=device)
    y_bar, x_bar = y_samples.mean(dim=1), x_samples.mean(dim=1)
    #####################################
    ## naive_r
    r_samples_naive = y_bar / x_bar
    empirical_mean, empirical_std = torch.mean(r_samples_naive), torch.std(r_samples_naive)
    r_samples_naive_cpu = r_samples_naive.cpu().numpy()
    mu_r = mu_r.cpu().item()
    var_r = var_r.cpu().item()
    print("Analyzing naive_r ...")
    print(f"Empirical mean: {empirical_mean:.4f}, Theoretical mean: {mu_r:.4f}")
    print(f"Empirical std: {empirical_std:.4f}, Theoretical std: {np.sqrt(var_r):.4f}")
    # where to save fig
    root, base_name=os.path.dirname(results['prediction_file']), os.path.basename(results['prediction_file'])
    root = os.path.join(root, "naive_r_fit_for_gaussian")
    os.makedirs(root, exist_ok=True)
    plot_gaussian_distr(r_samples_naive_cpu, mu_r, var_r, os.path.join(root,base_name.replace(".nii.gz","_down100_10k.png")))
    # dict
    results['ratio']['naive_ratio']['gaussian_mu_y'], results['ratio']['gaussian_var_y']= mu_y, var_y
    results['ratio']['naive_ratio']['gaussian_mu_x'], results['ratio']['gaussian_var_x']= mu_x, var_x
    results['ratio']['naive_ratio']['gaussian_cov_xy'] = cov_x_y
    results['ratio']['naive_ratio']['gaussian_mu_r'], results['ratio']['gaussian_var_r'] = mu_r, var_r
    #####################################
    ## 2_order_corr_r
    x2_bar = (x_samples**2).mean(dim=1)
    y2_bar = (y_samples**2).mean(dim=1)
    cov_x_y = torch.mean((x_samples - x_bar[:, None]) * (y_samples - y_bar[:, None]), dim=1)
    cov_x2_y = torch.mean((x_samples ** 2 - x2_bar[:, None]) * (y_samples - y_bar[:, None]), dim=1)
    cov_y2_x = torch.mean((y_samples ** 2 - y2_bar[:, None]) * (x_samples - x_bar[:, None]), dim=1)
    cov_x2_x = torch.mean((x_samples ** 2 - x2_bar[:, None]) * (x_samples - x_bar[:, None]), dim=1)
    var_x = x_samples.var(dim=1)
    var_y = y_samples.var(dim=1)
    ## r_a* 和 r_b*
    r_a = cov_x_y / (x_bar * y_bar)
    r_a_star = r_a * (1 + (1 / (n - 1)) * ((y_bar * cov_x2_y + x_bar * cov_y2_x) / (cov_x_y * x_bar * y_bar) - 4) - (
                1 / (n - 1)) * (var_x / x_bar ** 2 + var_y / y_bar ** 2 + 2 * cov_x_y / (x_bar * y_bar)))

    r_b = var_x / x_bar ** 2
    r_b_star = r_b * (1 + (4 / (n - 1)) * ((0.5 * cov_x2_x) / (x_bar * var_x) - 1) - (4 / (n - 1)) * (var_x / x_bar ** 2))
    r_samples_second_corr = r_samples_naive * (1 - (1 / n) * (r_b_star - r_a_star) - (1 / n ** 2) * (
                (cov_x2_y - 2 * x_bar * cov_x_y) / (x_bar ** 2 * y_bar) - (cov_x2_x - 2 * x_bar * var_x) / (x_bar ** 3) - (
                    3 * var_x * cov_x_y) / (x_bar ** 3 * y_bar) + (3 * var_x ** 2) / (x_bar ** 4)))
    # above, eqn (appendix C)
    empirical_mean, empirical_std = torch.mean(r_samples_second_corr), torch.std(r_samples_second_corr)
    r_samples_second_corr_cpu = r_samples_second_corr.cpu().numpy()
    print("Analyzing second_corr_r ...")
    print(f"Empirical mean: {empirical_mean:.4f}, Theoretical mean: {mu_r:.4f}")
    print(f"Empirical std: {empirical_std:.4f}, Theoretical std: {np.sqrt(var_r):.4f}")
    # where to save fig
    root, base_name=os.path.dirname(results['prediction_file']), os.path.basename(results['prediction_file'])
    root = os.path.join(root, "second_corr_r_fit_for_gaussian")
    os.makedirs(root, exist_ok=True)
    plot_gaussian_distr(r_samples_second_corr_cpu, mu_r, var_r, os.path.join(root,base_name.replace(".nii.gz","_down100_10k.png")))

    #####################################
    # dict
    results['ratio']['second_corr_ratio']['gaussian_mu_y'], results['ratio']['gaussian_var_y']= mu_y, var_y
    results['ratio']['second_corr_ratio']['gaussian_mu_x'], results['ratio']['gaussian_var_x']= mu_x, var_x
    results['ratio']['second_corr_ratio']['gaussian_cov_xy'] = cov_x_y
    results['ratio']['second_corr_ratio']['gaussian_mu_r'], results['ratio']['gaussian_var_r'] = mu_r, var_r
    asd
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

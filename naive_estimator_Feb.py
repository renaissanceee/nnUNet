import pickle
import os
import numpy as np
import nibabel as nib

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


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

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:#  [(1,2,3), (2,3), (3,)]
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r) # e.g.(1,2,3)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask=None)# np (1,155,240,240)
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
def calc_wt_ratio(prob, gt):
    # (240,240,155)

    joint = prob * gt # elem-wise # max is 0.003
    # joint_mean = np.mean(joint_product)
    # gt_mean = np.mean(gt)
    
    # count_greater_than_0_5 = np.sum(prob > 0.5)
    # count_0 = np.sum(np.isclose(gt, 1.0))
    # print(count_greater_than_0_5,count_0)
    print(np.sum((prob > 0.5) & (gt == 1.0)))
    import pdb;pdb.set_trace()


    sum_joint, sum_gt = np.sum(joint), np.sum(gt)
    wt_ratio = sum_joint/sum_gt
    print(f'joint, gt: {sum_joint}, {sum_gt}')
    print(f'wt_ratio: {wt_ratio}')
    return wt_ratio

preds_root = '/staging/leuven/stg_00081/jli/calibration/nnUNet/nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_50epochs/fold_0/'
file_name = 'BraTS2021_00000'
file_name = 'BraTS2021_00464'

############################################
# ## pkl is a dict()##
# pkl_path = os.path.join(preds_rootroot, file_name+'.pkl')
# with open(pkl_path, "rb") as f:
#     pkl_data = pickle.load(f)
# # sitk_stuff: {'spacing': (1.0, 1.0, 1.0), 'origin': (-0.0, -239.0, 0.0),
# #              'direction': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)}
# # spacing: [1.0, 1.0, 1.0]
# # shape_before_cropping: (155, 240, 240)
# # bbox_used_for_cropping: [[4, 150], [40, 211], [52, 188]]
# # shape_after_cropping_and_before_resampling: (146, 171, 136)
############################################

## probabilities ##
npz_path = os.path.join(preds_root, file_name+'.npz')
probs = np.load(npz_path)
probs = probs['probabilities']# (3,155,240,240)

## preds ##
nii_path = os.path.join(preds_root, file_name+'.nii.gz')
preds = nib.load(nii_path).get_fdata() # np, [240,240,155] belong to labels {0,1,2,3}, due  to overwritten
# print(f'preds: {preds.shape}')

## labels ##
label_root = '/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_preprocessed/Dataset137_BraTS2021/gt_segmentations/'
# label_root = '/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/Dataset137_BraTS2021/labelsTs/fold_0/' # equal as bebore
nii_path = os.path.join(label_root, file_name+'.nii.gz')
label = nib.load(nii_path).get_fdata() # [240,240,155]

## ratio-estimator ##
wt_label = np.clip(label, None, 1.0) # 1,2,3 -> 1
wt_probs = np.transpose(probs[0,:,:,:], (1, 2, 0)) # (240,240,155)

wt_ratio = calc_wt_ratio(wt_probs,wt_label)

# import pdb;pdb.set_trace()

# et_fraction = calc_et_fraction(preds)





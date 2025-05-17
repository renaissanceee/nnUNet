import numpy as np
import nibabel as nib
import os
import torch
import argparse
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join,subfiles
import torch.nn.functional as F
# from nnunetv2.evaluation.ece_utils import calc_label_shift
from nnunetv2.evaluation.ece_label_shift import get_importance_weights,EceLabelShift
from nnunetv2.evaluation.plot_utils import plot_ce_and_range_dataset, plot_bins_dataset,\
                                            plot_ratio_and_range_dataset, plot_ratio_and_range_all
from nnunetv2.evaluation.seg_utils import gather_files, region_or_label_to_mask, region_or_label_to_mask_prob_add

def convert_labels_to_one_hot(labels, num_classes=4):
    return F.one_hot(labels.long(), num_classes=num_classes).bool()  # 返回布尔类型

def cat_all_files(preds_source_list, preds_target_list, labels_source_list):
    print(f'concat for {len(preds_source_list)} preds_source ...')
    preds_list = []
    for file in preds_source_list:
        data = np.load(file)['probabilities']  # (4, 155, 240, 240)
        data = torch.from_numpy(data).float()  # to Tensor
        data = data.permute(1, 2, 3, 0).reshape(-1, 4)  # → (155, 240, 240, 4)
        preds_list.append(data)
    preds_source = torch.cat(preds_list, dim=0)

    print(f'concat for {len(preds_target_list)} preds_target ...')
    preds_list = []
    for file in preds_target_list:
        data = np.load(file)['probabilities']  # (4, 155, 240, 240)
        data = torch.from_numpy(data).float()  # to Tensor
        data = data.permute(1, 2, 3, 0).reshape(-1, 4)  # → (155, 240, 240, 4)
        preds_list.append(data)
    preds_target = torch.cat(preds_list, dim=0)

    print(f'concat for {len(labels_source_list)} labels_source ...')
    preds_list = []
    for file in labels_source_list:
        data = nib.load(file).get_fdata()
        data = torch.from_numpy(data).float()
        data = data.reshape(-1)
        preds_list.append(data)
    labels_source = torch.cat(preds_list, dim=0)  # cat for (N * 155*240*240)

    return preds_source, preds_target, labels_source

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, required=True, help="CELoss")
    parser.add_argument("--fold", type=str, required=True, help="0")
    parser.add_argument('--biomarker', required=True, type=str, help='ntr, ctr')
    args = parser.parse_args()

    pred_root = "/staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021"
    label_root = os.environ.get("nnUNet_raw")
    preds_val_dir = join(pred_root,f"nnUNetTrainer{args.loss}__nnUNetPlans__2d",f"fold_{args.fold}","validation_ece")
    preds_test_dir = preds_val_dir.replace('validation_ece', 'test')
    labels_val_dir = join(label_root, f"Dataset137_BraTS2021/labelsVal_ece/fold_{args.fold}")

    preds_source_list = subfiles(preds_val_dir, suffix='.npz', join=True)
    preds_target_list = subfiles(preds_test_dir, suffix='.npz', join=True)
    labels_source_list = subfiles(labels_val_dir, suffix='.nii.gz', join=True)
    n_volumes = len(preds_target_list)
    print(f"calc label_shift for {n_volumes} volumes in test-set")
    ########## to delete ##########
    # n_volumes = 2
    # preds_source_list, preds_target_list, labels_source_list = preds_source_list[:n_volumes], preds_target_list[:n_volumes], labels_source_list[:n_volumes]
    ########## to delete ##########

    preds_source, preds_target, labels_source = cat_all_files(preds_source_list, preds_target_list, labels_source_list)
    labels_source_one_hot = convert_labels_to_one_hot(labels_source) #[N]->[N,4]
    output = get_importance_weights(preds_source.numpy(), labels_source_one_hot.numpy(), preds_target.numpy())  # prob [N,4]
    weights = torch.tensor(output["weights"]) ## print("four class weights: ", weights)


    interested_region = (2,) if args.biomarker == 'ntr' else (2, 3)
    # GT: nec, wt
    nec_gt_map_source, _ = region_or_label_to_mask(labels_source, interested_region)
    nec_gt_map_source = torch.from_numpy(nec_gt_map_source)
    nec_prob_map_source = preds_source[:, list(interested_region)].sum(dim=1)
    # prob: nec, wt
    nec_prob_map_target = preds_target[:, list(interested_region)].sum(dim=1)
    # weight
    nec_weight = weights[list(interested_region)].sum(dim=0)

    estimator = EceLabelShift(adaptive_bins=True, n_bins=15, p=1, classwise=True)
    nec_ece_all = estimator(
        preds_target=nec_prob_map_target,
        preds_source=nec_prob_map_source,
        labels_source=nec_gt_map_source,
        weights=nec_weight,  # for necrosis
        n_volumes = n_volumes
    )
    import pdb;pdb.set_trace()
    
    ## source
    # nec_prob_map_source = preds_source[:, list(interested_region)].sum(dim=1)
    # nec_gt_map_source, _ = region_or_label_to_mask(labels_source, interested_region)
    # nec_gt_map_source = torch.from_numpy(nec_gt_map_source)
    
    


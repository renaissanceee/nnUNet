import numpy as np
import os
import joblib
import argparse
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from batchgenerators.utilities.file_and_folder_operations import subfiles

def apply_isotonic_calibration(calibrator_dir, test_input_dir, test_output_dir, num_classes=4):
    os.makedirs(test_output_dir, exist_ok=True)

    calibrators = []
    for c in range(num_classes):
        calibrator_path = os.path.join(calibrator_dir, f"isotonic_calibrator_class{c}.pkl")
        calibrators.append(joblib.load(calibrator_path))

    # 获取所有待处理的 .npz 文件
    test_files = subfiles(test_input_dir, suffix='.npz', join=True)

    for npz_path in tqdm(test_files, desc=f"Calibrating {len(test_files)} files"):
        data = np.load(npz_path)['probabilities']  # shape: (C, 155, 240, 240)
        calibrated = np.zeros_like(data)
        for c in range(num_classes):
            class_prob = data[c]
            flat = class_prob.reshape(-1, 1)
            # scaled = MinMaxScaler().fit_transform(flat)
            scaled = flat
            calibrated_flat = calibrators[c].transform(scaled)
            calibrated[c] = calibrated_flat.reshape(class_prob.shape)

        out_path = os.path.join(test_output_dir, os.path.basename(npz_path))
        np.savez_compressed(out_path, probabilities=calibrated)

# def apply_isotonic_calibration(calibrator_dir, test_input_dir, test_output_dir, num_classes=4):
#     os.makedirs(test_output_dir, exist_ok=True)
#
#     calibrators = []
#     for c in range(num_classes):
#         calibrator_path = os.path.join(calibrator_dir, f"isotonic_calibrator_class{c}.pkl")
#         calibrators.append(joblib.load(calibrator_path))
#
#     test_files = subfiles(test_input_dir, suffix='.npz', join=True)
#     all_probs = [[] for _ in range(num_classes)]
#     shapes = []
#
#     # Step 1: collect and flatten all probabilities for each class
#     for npz_path in tqdm(test_files, desc="Loading and stacking"):
#         data = np.load(npz_path)['probabilities']  # shape: (C, H, W, D)
#         shapes.append((npz_path, data.shape))  # Keep track of shape for restoring later
#         for c in range(num_classes):
#             flat = data[c].reshape(-1, 1)
#             scaled = MinMaxScaler().fit_transform(flat)  # Scaling before calibration
#             all_probs[c].append(scaled)
#
#     # Step 2: concatenate all data per class and apply isotonic transform
#     all_calibrated = []
#     for c in range(num_classes):
#         class_probs = np.vstack(all_probs[c])  # shape: (N, 1)
#         calibrated_probs = calibrators[c].transform(class_probs)
#         all_calibrated.append(calibrated_probs)
#
#     # Step 3: split calibrated data and save per original file
#     idx = 0
#     for npz_path, shape in tqdm(shapes, desc="Saving calibrated outputs"):
#         _, C, H, W = shape
#         N_voxels = H * W * C
#
#         calibrated = np.zeros((num_classes, H, W, C), dtype=np.float32)
#         for c in range(num_classes):
#             calibrated_slice = all_calibrated[c][idx:idx + N_voxels]
#             calibrated[c] = calibrated_slice.reshape((H, W, C))
#
#         idx += N_voxels
#         out_path = os.path.join(test_output_dir, os.path.basename(npz_path))
#         np.savez_compressed(out_path, probabilities=calibrated)
#
#     # 清空缓存
#     import gc
#     gc.collect()
#     try:
#         import torch
#         torch.cuda.empty_cache()
#     except ImportError:
#         pass


def main(fold: int, loss_name: str):
    base_path = "/staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021"
    trainer = f"nnUNetTrainer{loss_name}__nnUNetPlans__2d"
    calibrator_dir = os.path.join(base_path, trainer, f"fold_{fold}")

    ## val
    val_input_dir = os.path.join(calibrator_dir, "validation_ece")
    val_output_dir = os.path.join(calibrator_dir, "validation_ece_IR")
    os.makedirs(val_output_dir, exist_ok=True)
    apply_isotonic_calibration(calibrator_dir, val_input_dir, val_output_dir)

    ## test
    test_input_dir = os.path.join(calibrator_dir, "test")
    test_output_dir = os.path.join(calibrator_dir, "test_IR")
    os.makedirs(test_output_dir, exist_ok=True)
    apply_isotonic_calibration(calibrator_dir, test_input_dir, test_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number (e.g. 0-4)")
    parser.add_argument("--loss", type=str, required=True, help="Loss name (e.g. CELoss, DiceLoss)")
    args = parser.parse_args()

    main(args.fold, args.loss)

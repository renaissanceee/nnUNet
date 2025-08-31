import os
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import shutil
import sys
from copy import deepcopy
from typing import Tuple, Union, List
import torch.nn.functional as F
import numpy as np
import torch
import random


def copy_brats_from_Tr_to_Ts_or_Val(val_keys, set_train_folder_img, set_train_folder_label,
                                set_test_folder_img, set_test_folder_label, modalities=4):
    """
    Params:
        val_keys (list): 验证集病例ID列表 (如 ['BraTS2021_00000', ...])
        set_train_folder_img (str): 原始训练图像文件夹路径
        set_train_folder_label (str): 原始训练标签文件夹路径
        set_test_folder_img (str): 目标测试图像文件夹路径
        set_test_folder_label (str): 目标测试标签文件夹路径
        modalities (int): 模态数量 (默认为4)
    """
    os.makedirs(set_test_folder_img, exist_ok=True)
    os.makedirs(set_test_folder_label, exist_ok=True)

    for case_id in val_keys:  # ~ volume
        for mod in range(modalities):  # ~ images(4-modalities)
            src_img = join(set_train_folder_img, f"{case_id}_000{mod}.nii.gz")
            dst_img = join(set_test_folder_img, f"{case_id}_000{mod}.nii.gz")
            shutil.copy(src_img, dst_img)

        src_label = join(set_train_folder_label, f"{case_id}.nii.gz") # ~labels
        dst_label = join(set_test_folder_label, f"{case_id}.nii.gz")
        shutil.copy(src_label, dst_label)


def split_nested_keys_TS(tr_keys, val_ratio_TS=0.1, val_ratio_ece=0.1, seed=12345):
    random.seed(seed)
    keys = tr_keys.copy()
    random.shuffle(keys)
    val_size = int(len(keys) * (val_ratio_TS+val_ratio_ece)) # 0.1+0.1
    val_TS_size = int(len(keys) * val_ratio_TS)# 0.1
    val_keys = keys[:val_size]
    # val_keys=val_TS_keys+val_ece_keys
    val_TS_keys, val_ece_keys = val_keys[:val_TS_size],val_keys[val_TS_size:]
    new_tr_keys = keys[val_size:]
    return new_tr_keys, val_TS_keys, val_ece_keys

def split_nested_keys(tr_keys, val_ratio=0.1, seed=12345):
    random.seed(seed)
    keys = tr_keys.copy()
    random.shuffle(keys)
    val_size = int(len(keys) * val_ratio)  # 0.1
    new_tr_keys, val_keys = keys[val_size:], keys[:val_size]
    return new_tr_keys, val_keys
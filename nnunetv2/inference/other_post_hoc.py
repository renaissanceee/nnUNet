import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler
# from dirichletcal import DirichletCalibrator
from sklearn.model_selection import train_test_split
import joblib
import torch
# def get_scores_after_isotonic(scores_val, labels_val, scores_test):
#     calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
#     scaler = MinMaxScaler()
#     scaled_scores_val = scaler.fit_transform(scores_val)
# 
#     calibrated_scores_test = np.zeros_like(scores_test)
#     for class_idx in range(scores_test.shape[1]):
#         mask = (labels_val == class_idx)
#         calibrator.fit(scaled_scores_val[:, class_idx], mask)
#         calibrated_scores_test[:, class_idx] = calibrator.transform(scores_test[:, class_idx])
# 
#     calibrated_scores_test = scaler.inverse_transform(calibrated_scores_test)
#     calibrated_scores_test = torch.clamp(torch.tensor(calibrated_scores_test), min=EPS, max=1 - EPS)
#     return calibrated_scores_test


def get_scores_after_isotonic(scores_val, labels_val, save_path):
    scores_val = torch.softmax(scores_val.float(), dim=1)
    scores_val = scores_val.cpu().numpy()
    labels_val = labels_val.cpu().numpy()

    N, num_classes = scores_val.shape[0], scores_val.shape[1]
    scaler = MinMaxScaler()# 归一化
    scaled_scores_val = scaler.fit_transform(scores_val)
    # scaled_scores_test = scaler.transform(scores_test)
    # calibrated_scores_test = np.zeros_like(scores_test)

    # indices = np.random.choice(N, size=10000000, replace=False)# downsample
    # scores_val_sampled = scores_val[indices]
    # labels_val_sampled = labels_val[indices]
    # import pdb;pdb.set_trace()
    for class_idx in range(num_classes):
        y_score = scaled_scores_val[:, class_idx]# 获取每个样本在该类别下的概率和标签（one-vs-rest）
        y_true = (labels_val == class_idx).astype(int)

        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        calibrator.fit(y_score, y_true)
        joblib.dump(calibrator, join(save_path, f'isotonic_calibrator_class{class_idx}.pkl'))


    # 对 test 分数做校准
    # calibrator = joblib.load(join(save_path, f'isotonic_calibrator_class{class_idx}.pkl'))
    # calibrated_scores_test[:, class_idx] = calibrator.predict(scaled_scores_test[:, class_idx])
    return calibrator


# def get_scores_after_direchlet(scores_val, labels_val, scores_test):
#     y_prob_flat = scores_val.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [N*H*W, C]
#     y_true_flat = labels_val.reshape(-1)                                   # [N*H*W]
#
#     # 可选：采样子集避免 OOM
#     x_calib, _, y_calib, _ = train_test_split(
#         y_prob_flat.numpy(), y_true_flat.numpy(), train_size=100000, stratify=y_true_flat.numpy()
#     )
#
#     calibrator = DirichletCalibrator(reg_lambda=1e-3, reg_mu=1e-1)
#     calibrator.fit(x_calib, y_calib)
#     # 假设你现在有新的测试集输出 y_prob_test: [N, C, H, W]
#     y_prob_test = scores_test.permute(0, 2, 3, 1).reshape(-1, num_classes).numpy()
#     calibrated_scores_test = calibrator.transform(y_prob_test)
#     calibrated_scores_test = torch.tensor(y_prob_calibrated).reshape(N, H, W, C).permute(0, 3, 1, 2)
#     return calibrated_scores_test

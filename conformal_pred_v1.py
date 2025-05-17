import numpy as np
import nibabel as nib
import os
import torch
import argparse
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, subfiles,save_json, load_json,isfile
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import optimize
from scipy.special import rel_entr

def logit(x):
    eps = 1e-6
    x = np.clip(x, eps, 1 - eps)
    return np.log(x) - np.log(1 - x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def detect_failure(lower, upper, test_r_gt, case_ids):
    mask = (test_r_gt < lower) | (test_r_gt > upper)
    failure = [case_ids[i] for i in np.where(mask)[0]]
    return failure

def plot_conformal(test_r_est, test_r_gt, intervals, case_ids, save_path="conformal_intervals.png"):
    font_size, name_size = 22, 20
    x = np.arange(len(test_r_est))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor('#e6e9f0')  # 设置背景色

    # 自定义rectangle
    for i, (l, u) in enumerate(intervals):
        ax.plot([i, i], [l, u], color='#d1a3e3', linewidth=5)

    ax.scatter(x, test_r_est,
               color='#ba68c8',
               label=r'$r$',
               zorder=3,
               s=100)

    # 绘制真实值（散点），略偏右
    ax.scatter(x + 0.15, test_r_gt,
               color='blue',
               label=r'$r_{gt}$',
               zorder=4,
               s=80)

    # 去除四条边框
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # 设置标题和坐标轴
    ax.set_title("Conformal Prediction Intervals", fontsize=font_size, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=40, ha='center', fontsize=int(0.8*name_size))
    ax.tick_params(axis='x', labelsize=int(0.8*name_size))
    ax.tick_params(axis='y', labelsize=name_size)
    ax.set_xlabel("Volume", fontsize=font_size)
    ax.set_ylabel("Ratio", fontsize=font_size)

    handles, labels = ax.get_legend_handles_labels()

    # 自定义
    # ice_handle = Rectangle((0, 0), width=0.3, height=0.1,color='#d1a3e3', alpha=0.5)
    ########################################
    ice_handle = Rectangle(
        (0, 0),  # x, y 坐标（不影响图例）
        width=0.3,  # 宽度：调大就更粗
        height=1.1,  # 高度：看起来像竖线
        color='#d1a3e3',
        alpha=0.5,
    )
    handles, labels = ax.get_legend_handles_labels()
    # r_gt → r → I_{CE}
    order = [labels.index(r'$r_{gt}$'), labels.index(r'$r$')]
    ordered_handles = [handles[i] for i in order] + [ice_handle]
    ordered_labels = [labels[i] for i in order] + [r'$I_{CP}$']
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=font_size,
        handleheight=1.1,  # 拉高 legend 中 handle 的高度
        handlelength=0.3  # 缩短横向长度，避免看起来横着
    )

    # 网格和纵轴范围
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# def binary_kl(pred_ratio, y):
#   return special.rel_entr(y, pred_ratio) + special.rel_entr(1-y, 1-pred_ratio)

def binary_kl(p, q):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return rel_entr(p, q) + rel_entr(1 - p, 1 - q)

# 一阶导数
def gradient_q_binary_kl(p, q):
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return np.log(q) - np.log(1 - q) - np.log(p) + np.log(1 - p)

# 二阶导数（修正了你之前写错的表达式）
def gradient_q_2_binary_kl(p, q):
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return 1 / q + 1 / (1 - q)

# 主函数：输入多个 pred_ratio，输出 [N, 2] 的 interval 数组
def conformal_interval_kl_div(pred_ratios, quantile_value):
    intervals = []

    for p in pred_ratios:
        init_left = p / 2
        init_right = (1 - p) / 2 + p

        def target_fun(y):
            f = binary_kl(p, y) - quantile_value
            f_prime = gradient_q_binary_kl(p, y)
            f_prime2 = gradient_q_2_binary_kl(p, y)
            return f, f_prime, f_prime2

        try:
            sol_left = optimize.root_scalar(
                lambda y: target_fun(y)[0],
                x0=init_left,
                fprime=lambda y: target_fun(y)[1],
                fprime2=lambda y: target_fun(y)[2],
                method='halley'
            )
            left = sol_left.root if sol_left.converged else 0
        except Exception:
            left = 0

        try:
            sol_right = optimize.root_scalar(
                lambda y: target_fun(y)[0],
                x0=init_right,
                fprime=lambda y: target_fun(y)[1],
                fprime2=lambda y: target_fun(y)[2],
                method='halley'
            )
            right = sol_right.root if sol_right.converged else 1
        except Exception:
            right = 1

        intervals.append([left, right])

    return np.array(intervals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, required=True, help="CELoss")
    parser.add_argument("--fold", type=str, required=True, help="0")
    parser.add_argument('--biomarker', required=False, default='ntr', type=str, help='ntr, ctr')
    parser.add_argument('--CP', required=False, default=80, type=int, help='80, 90')
    parser.add_argument('--TS', required=False, default=None, type=str, help='temperature scaling')
    args = parser.parse_args()

    # root = f"/staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainer{args.loss}__nnUNetPlans__2d/fold_{args.fold}"
    root = f"/scratch/leuven/372/vsc37255/nnUNetTrainer{args.loss}__nnUNetPlans__2d/fold_{args.fold}"
    val_dir = join(root, "validation_ece/ratio_metrics_prob_ntr")
    test_dir = join(root, "test/ratio_metrics_prob_ntr")
    folder_root = join(root, f"test/ratio_metrics_prob_ntr")
    if args.TS is not None:
        val_dir = val_dir.replace('validation_ece',f'validation_ece_TS_{args.TS}')
        test_dir = test_dir.replace('test',f'test_TS_{args.TS}')
        folder_root =folder_root.replace('test',f'test_TS_{args.TS}')
    folder_save = join(folder_root, f"conformal_{args.CP}")  # 'conformal_{args.CP}_kl'
    os.makedirs(folder_save, exist_ok=True)
    print('loading files ...')
    # test: r_gt, r_est
    test_json_path = join(test_dir, "plot/plot_bins15_ratio.json")
    test_json = load_json(test_json_path)
    test_r_est = test_json["r_naive"]["r_est"]
    test_r_gt = test_json["r_gt"]["r_gt"]
    name_list = subfiles(join(root, 'test'), suffix='.npz', join=False)
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in name_list]
    # val: r_gt, r_est
    val_json_path = join(val_dir, "bins15_ratio.json")
    val_json = load_json(val_json_path)
    val_r_est = np.array([per_case["ratio"]["r_naive"]["r_est"] for per_case in val_json["ratio_per_case"]])
    val_r_gt = np.array([per_case["ratio"]["r_gt"]["r_gt"] for per_case in val_json["ratio_per_case"]])

    "conformal_pred_v1"
    print('calc linear residual ...')
    residuals = np.abs(val_r_est - val_r_gt)
    q = np.quantile(residuals, args.CP/100)  # 90% conf_interval
    lower, upper = np.clip(test_r_est - q, 0.0, 1.0), np.clip(test_r_est + q, 0.0, 1.0) # clamp for [0,1]
    
    "conformal_pred_v2"  # log-mapping+sigmoid-->but wide
    # residuals = np.abs(logit(val_r_est) - logit(val_r_gt))
    # q = np.quantile(residuals, args.CP/100)  # 90% conf_interval
    # lower, upper = sigmoid(logit(test_r_est) - q), sigmoid(logit(test_r_est) + q)  # clamp for [0,1]

    "conformal_pred_v3"   # kl
    # kl_values = binary_kl(val_r_est, val_r_gt)
    # quantile_value = np.quantile(kl_values, args.CP/100)
    # interval = conformal_interval_kl_div(test_r_est, quantile_value)
    # lower, upper = interval[:,0],interval[:,1]

    "save_json"
    failure = detect_failure(lower,upper,test_r_gt,case_ids)  # fail_case: outside the range
    mean_r_range = np.mean(upper-lower)
    result = {'failure': failure, 'mean_r_range': mean_r_range,'interval_per_case':np.stack((lower, upper), axis=1).tolist()}
    save_json(result, join(folder_root, f"CP{args.CP}_ratio.json"), sort_keys=False) ## CP80_ratio.json
    print('Fail, Range:', len(failure), mean_r_range)
    if args.fold == '0':
        step_size = 10
        for start in range(0, len(test_r_est), step_size): #
            end = start + step_size
            intervals = list(zip(lower[start:end], upper[start:end]))
            plot_conformal(test_r_est[start:end], test_r_gt[start:end], intervals, case_ids[start:end], save_path=join(folder_save, f"conformal_interval_{end}.png"))





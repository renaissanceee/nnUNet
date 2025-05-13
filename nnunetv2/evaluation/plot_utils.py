import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def plot_histogram_bias(values, title, xlabel, save_path, color):
    # font_size, name_size = 18, 14
    font_size, name_size = 22, 20
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    bin_width = 0.02  # 设置 bin 宽度
    bins = np.arange(0, 0.2 + bin_width, bin_width)  # 按照宽度设置 bins
    plt.hist(values, bins=bins, color=color, edgecolor=None, #'black',
             weights=np.ones_like(values) / len(values) * 100)# 百分比
    ax.set_facecolor('#e6e9f0')  # 设置浅灰色背景

    # 去掉边框框线
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 0.2)
    plt.ylim(0, 60) # 70
    plt.xticks(np.arange(0, 0.21, 0.05))
    plt.yticks(np.arange(10, 51, 10))
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_histogram_range(values, title, xlabel, save_path, color):
    # font_size, name_size = 18, 14
    font_size, name_size = 22, 20
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    bin_width = 0.02  # 设置 bin 宽度
    bins = np.arange(0, 1 + bin_width, bin_width)  # 按照宽度设置 bins
    plt.hist(values, bins=bins, color=color, edgecolor=None, #'black',
             weights=np.ones_like(values) / len(values) * 100)# 百分比
    ax.set_facecolor('#e6e9f0')  # 设置浅灰色背景

    # 去掉边框框线
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 30) # 50
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(5, 30, 5))
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_corr_and_range(r_est_naive, r_est_second, r_gt, bound_naive, bound_second, save_path='plot.png', sigma="",
                     name_list=[]):
    gt_color = 'blue' # JJ
    naive_color = '#ba68c8'
    second_color = '#81c784' # 'darkseagreen' # 'green'

    # xtick_labels
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in
                name_list]  # "..."->"BraTS2021_00753.nii.gz"-> "00753"
    x = np.arange(len(r_gt))  # 根据 r_gt 的长度生成 x
    fig, ax = plt.subplots(figsize=(9, 5))
    ## Apr.23
    dist = 0.15 # 0.1
    # 绘制真实值（红色散点）
    ax.scatter(
        x,
        r_gt,
        color=gt_color,
        label=r'$r_{gt}$',
        zorder=4,
        s=80
    )
    # overall-bar: r_naive, r_second
    r_est_array = np.full_like(x, r_est_naive, dtype=float)  # extend dim
    ax.errorbar(
        x - dist,
        r_est_array,  # extend
        yerr=[r_est_array - bound_naive[:, 0], bound_naive[:, 1] - r_est_array],
        fmt='o',
        markersize=10,
        markeredgewidth=2,
        markerfacecolor=naive_color,
        markeredgecolor=naive_color,
        color=naive_color,
        capsize=8,
        capthick=15,
        linewidth=5,
        zorder=3,
        label=r'$r$'
    )
    r_est_array = np.full_like(x, r_est_second, dtype=float)  # extend dim
    ax.errorbar(
        x + dist,
        r_est_array,  # extend
        yerr=[r_est_array - bound_second[:, 0], bound_second[:, 1] - r_est_array],
        fmt='o',
        markersize=10,
        markeredgewidth=2,
        markerfacecolor=second_color,
        markeredgecolor=second_color,
        color=second_color,
        capsize=8,
        capthick=15,
        linewidth=5,
        zorder=3,
        label=r'$r_{corr,2}$'
    )

    # font_size, name_size = 18, 14
    font_size, name_size = 22, 20
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=40, ha='center', fontsize=int(0.8*name_size))
    ax.set_facecolor('#e6e9f0') 
    ax.set_xlabel('Volume', fontsize=font_size)
    ax.set_ylabel('Ratio', fontsize=font_size)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=font_size)  # outside

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    # ax.set_title(f'Ratio and Confidence Interval(±{sigma}$\\sigma$)', fontsize=font_size)
    ax.set_title(f'Debiased Effect on 10 Volumes (±{sigma}$\\sigma$)', fontsize=font_size, pad=15)
    # 去掉边框框线
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    # 分别设置 x/y 轴刻度字体大小
    ax.tick_params(axis='x', labelsize=int(0.8*name_size))
    ax.tick_params(axis='y', labelsize=name_size)
    # import pdb;pdb.set_trace()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ce_and_range(r_est, r_gt, bound_ce, bound, save_path='plot.png', sigma="", name_list=[]):
    gt_color = 'blue'
    ce_color = '#1565c0' # '#90caf9' # 'lightseagreen' # 'coral' # 'orange' # JJ
    overall_color = '#ba68c8' # 'darkseagreen' # 'green'

    # xtick_labels
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in
                name_list]  # "..."->"BraTS2021_00753.nii.gz"-> "00753"
    x = np.arange(len(r_gt))  # 根据 r_gt 的长度生成 x
    fig, ax = plt.subplots(figsize=(9, 5))
    ## Apr.23
    dist = 0.15 # 0.1
    # 扩展 r_est 为数组（所有点相同）
    r_est_array = np.full_like(x, r_est, dtype=float)
    # 绘制真实值（红色散点）
    ax.scatter(
        x,
        r_gt,
        color=gt_color,
        label=r'$r_{gt}$',
        zorder=4,
        s=80
    )
    # overall-bar（用 bound）
    ax.errorbar(
        x - dist,
        r_est_array,
        yerr=[r_est_array - bound[:, 0], bound[:, 1] - r_est_array],
        fmt='o',
        markersize=10,
        markeredgewidth=2,
        markerfacecolor=overall_color,
        markeredgecolor=overall_color,
        color=overall_color,
        capsize=8, # 10,
        capthick=15,
        linewidth=5,
        zorder=3,
        label=r'$r$'
    )
    # ce-区间块（用 bound_ce）
    ax.vlines(
        x + dist,
        bound_ce[:, 0],
        bound_ce[:, 1],
        color=ce_color,
        linewidth=8,
        alpha=0.5,
        zorder=2,
        label=r'$I_{CE}$'
    )

    # font_size, name_size = 18, 14
    font_size, name_size = 22, 20

    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=40, ha='center', fontsize=int(0.8*name_size))
    ax.set_facecolor('#e6e9f0')  
    ax.set_xlabel('Volume', fontsize=font_size)
    ax.set_ylabel('Ratio', fontsize=font_size)
    ########################################
    ice_handle = Rectangle(
        (0, 0),  # x, y 坐标（不影响图例）
        width=0.3,  # 宽度：调大就更粗
        height=1.1,  # 高度：看起来像竖线
        color=ce_color,
        alpha=0.5,
    )
    handles, labels = ax.get_legend_handles_labels()
    # r_gt → r → I_{CE}
    order = [labels.index(r'$r_{gt}$'), labels.index(r'$r$')]
    ordered_handles = [handles[i] for i in order] + [ice_handle]
    ordered_labels = [labels[i] for i in order] + [r'$I_{CE}$']
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=font_size,
        handleheight=1.1,  # 拉高 legend 中 handle 的高度
        handlelength=0.3  # 缩短横向长度，避免看起来横着
    )

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f'Miscalibration on 10 Volumes (±{sigma}$\\sigma$)', fontsize=font_size, pad=15)
    # 去掉边框框线
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    # 分别控制 x 和 y 轴的刻度标签字体大小
    ax.tick_params(axis='x', labelsize=int(0.8*name_size))
    ax.tick_params(axis='y', labelsize=name_size)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ratio_and_range(r_est, r_gt, bound, save_path, sigma="", name_list=[]):
    gt_color = 'blue'
    overall_color = '#ba68c8' # 'darkseagreen' # 'green'

    # xtick_labels
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in
                name_list]  # "..."->"BraTS2021_00753.nii.gz"-> "00753"
    x = np.arange(len(r_gt))  # 根据 r_gt 的长度生成 x
    fig, ax = plt.subplots(figsize=(9, 5))
    dist = 0.15 # 0.1
    r_est_array = np.full_like(x, r_est, dtype=float)
    # 绘制真实值（红色散点）
    ax.scatter(
        x,
        r_gt,
        color=gt_color,
        label=r'$r_{gt}$',
        zorder=4,
        s=80
    )
    # overall-bar（用 bound）
    ax.errorbar(
        x - dist,
        r_est_array,
        yerr=[r_est_array - bound[:, 0], bound[:, 1] - r_est_array],
        fmt='o',
        markersize=10,
        markeredgewidth=2,
        markerfacecolor=overall_color,
        markeredgecolor=overall_color,
        color=overall_color,
        capsize=8, # 10,
        capthick=15,
        linewidth=5,
        zorder=3,
        label=r'$r$'
    )

    font_size, name_size = 22, 20

    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=40, ha='center', fontsize=int(0.8*name_size))
    ax.set_facecolor('#e6e9f0')  
    ax.set_xlabel('Volume', fontsize=font_size)
    ax.set_ylabel('Ratio', fontsize=font_size)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f'Miscalibration on 10 Volumes (±{sigma}$\\sigma$)', fontsize=font_size, pad=15)
    # 去掉边框框线
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    # 分别控制 x 和 y 轴的刻度标签字体大小
    ax.tick_params(axis='x', labelsize=int(0.8*name_size))
    ax.tick_params(axis='y', labelsize=name_size)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ratio_and_range_all(paired_samples, folder_save, sigma, ce_type,ece_percentage):
    folder_save = join(folder_save, f"{ce_type}_plots_all", f"CE_{sigma}sigma__ratio_all")
    if ce_type=="kde":
        folder_save = join(folder_save, f"{ce_type}_plots_all_1e4", f"CE_{sigma}sigma__ratio_all") # 1e4
    os.makedirs(folder_save, exist_ok=True)
    if ece_percentage is not None:
        ece_percentage = f'_tile_{ece_percentage}'
    else:
        ece_percentage = ''
    save_path = join(folder_save, f"ratio_and_range_all{ece_percentage}.png")

    r_gt = paired_samples['r_gt']['r_gt']
    # name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']

    gt_color, bound_color = 'blue', '#ba68c8'
    font_size, name_size = 22, 20
    ## only 125 samples, otherwise too much/crowded.
    r_gt, bound_naive = r_gt[:len(r_gt)//2], bound_naive[:len(r_gt)//2, :]
    
    x = np.arange(len(r_gt))
    lower_err,upper_err  = r_gt - bound_naive[:, 0],bound_naive[:, 1] - r_gt
    # lower_err, upper_err = np.minimum(r_gt, r_gt - bound_naive[:, 0]), np.minimum(1 - r_gt, bound_naive[:, 1] - r_gt)
    lower_err, upper_err= np.clip(lower_err, 0, r_gt), np.clip(upper_err, 0, 1 - r_gt)
    yerr = np.vstack((lower_err, upper_err))  # shape [2, N]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, r_gt, yerr=yerr, fmt='o', color=gt_color,
                ecolor=bound_color, alpha=0.9, capsize=3)
    ax.set_facecolor('#e6e9f0')

    # 标签和标题
    ax.set_xlabel('Sample Index', fontsize=font_size)
    ax.set_ylabel('Ratio Estimation', fontsize=font_size)
    ax.set_title('Estimated Ratios with Confidence Intervals', fontsize=font_size)

    ax.set_xticks(x)
    # ax.set_xticklabels(name_list, rotation=45, ha='right', fontsize=int(0.8 * name_size))# 设置 x 轴标签为 name_list（如果提供）
    ax.set_ylim(0, 1)

    # 美化：去除边框
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # 坐标轴刻度大小
    ax.tick_params(axis='x', labelsize=int(0.8 * name_size))
    ax.tick_params(axis='y', labelsize=name_size)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_corr_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10, ce_type="bins", ece_percentage=None):
    if ece_percentage is not None:
        ece_percentage = f'_tile_{ece_percentage}'
    else:
        ece_percentage = ''
    folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}", f"CE_{sigma}sigma__corr")
    if ce_type=="kde":
        folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}_1e4", f"CE_{sigma}sigma__corr") # 1e4

    os.makedirs(folder_save, exist_ok=True)

    r_est_naive = paired_samples['r_naive']['r_est']
    r_est_second = paired_samples['r_second_corr']['r_est']
    r_gt = paired_samples['r_gt']['r_gt']
    name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']
    bound_second = paired_samples['r_second_corr'][f'bound__ce+{sigma_c}std']
    for start in range(0, len(r_est_naive), step_size):
        end = start + step_size
        save_path = join(folder_save, f"r_and_range_{end}.png")
        plot_corr_and_range(r_est_naive[start:end], r_est_second[start:end], r_gt[start:end], bound_naive[start:end],
                         bound_second[start:end], save_path, sigma, name_list[start:end])


def plot_ce_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10, ce_type="bins", ece_percentage=None):
    # if ce_type=="kde":
    #     folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}_1e4", f"CE_{sigma}sigma__ce") # 1e4
    # else:
    #     folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}", f"CE_{sigma}sigma__ce")
    if ece_percentage is not None:
        ece_percentage = f'_tile_{ece_percentage}'
    else:
        ece_percentage = ''
    folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}", f"CE_{sigma}sigma__ce")
    if ce_type=="kde":
        folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}_1e4", f"CE_{sigma}sigma__ce") # 1e4

    os.makedirs(folder_save, exist_ok=True)

    r_est_naive = paired_samples['r_naive']['r_est']
    r_gt = paired_samples['r_gt']['r_gt']
    name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']
    bound_ce_naive = paired_samples['r_naive']['bound__ce']
    for start in range(0, len(r_est_naive), step_size):
        end = start + step_size
        save_path = join(folder_save, f"ce_and_range_{end}.png")
        plot_ce_and_range(r_est_naive[start:end], r_gt[start:end], bound_ce_naive[start:end], bound_naive[start:end],
                          save_path, sigma, name_list[start:end])


def plot_ratio_and_range_dataset(paired_samples, folder_save, sigma="", step_size=10, ce_type="bins", ece_percentage=None):
    # if ce_type=="kde":
    #     folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}_1e4", f"CE_{sigma}sigma__ratio") # 1e4
    # else:
    #     folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}", f"CE_{sigma}sigma__ratio")

    if ece_percentage is not None:
        ece_percentage = f'_tile_{ece_percentage}'
    else:
        ece_percentage = ''
    folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}", f"CE_{sigma}sigma__ratio")
    if ce_type=="kde":
        folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}_1e4", f"CE_{sigma}sigma__ratio") # 1e4

    os.makedirs(folder_save, exist_ok=True)

    r_est_naive = paired_samples['r_naive']['r_est']
    r_gt = paired_samples['r_gt']['r_gt']
    name_list = paired_samples['reference_file']
    sigma_c = 1 if sigma == '' else sigma
    bound_naive = paired_samples['r_naive'][f'bound__ce+{sigma_c}std']

    for start in range(0, len(r_est_naive), step_size):
        end = start + step_size
        save_path = join(folder_save, f"r_and_range_{end}.png")
        plot_ratio_and_range(r_est_naive[start:end], r_gt[start:end], bound_naive[start:end], save_path, sigma, name_list[start:end])

def plot_bins_dataset(paired_samples, folder_save, sigma="",ce_type='bins', ece_percentage=None):
    # if ce_type=="kde":
    #     folder_save = join(folder_save, f"{ce_type}_hist_of_bias_and_range_1e4") # 1e4
    # else:
    #     folder_save = join(folder_save, f"{ce_type}_hist_of_bias_and_range")

    if ece_percentage is not None:
        ece_percentage = f'_tile_{ece_percentage}'
    else:
        ece_percentage = ''
    # folder_save = join(folder_save, f"{ce_type}_plots_step{step_size}{ece_percentage}", f"CE_{sigma}sigma__ratio")
    folder_save = join(folder_save, f"{ce_type}_hist_of_bias_and_range")
    if ce_type=="kde":
        folder_save = join(folder_save, f"{ce_type}_hist_of_bias_and_range_1e4") # 1e4


    os.makedirs(folder_save, exist_ok=True)

    sigma_c = 1 if sigma=="" else sigma
    range_ce = paired_samples['r_naive'][f'range__ce+{sigma_c}std']
    bias = paired_samples['r_naive']['bias_r']
    save_path_bias = join(folder_save, f"bias_hist_{sigma}sigma.png")
    plot_histogram_bias(bias, title=f'Overall Ratio Bias (±{sigma}$\\sigma$)',   # Bias
               xlabel='Ratio Bias', save_path=save_path_bias, color='skyblue')
    save_path_range = join(folder_save, f"range_hist_{sigma}sigma{ece_percentage}.png")

    ## Apr.23
    range_color = '#ba68c8' # 'darkseagreen' # 'limegreen'
    plot_histogram_range(range_ce, title=f'Overall Confidence Interval (±{sigma}$\\sigma$)', # Interval
                   xlabel=f'Interval Widthth', save_path=save_path_range, color=range_color)


import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import os

def plot_histogram_bias(values, title, xlabel, save_path, color):
    font_size, name_size = 18, 14
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    # plt.hist(
    #     values, bins=30, color=color, edgecolor='black', # 用30 bins表达
    #     weights=np.ones_like(values) / len(values) * 100
    # )
    bin_width = 0.02  # 设置 bin 宽度
    bins = np.arange(0, 0.2 + bin_width, bin_width)  # 按照宽度设置 bins
    plt.hist(values, bins=bins, color=color, edgecolor=None, #'black',
             weights=np.ones_like(values) / len(values) * 100)# 百分比
    ax.set_facecolor('#e6e9f0')  # 设置浅灰色背景

    # 去掉边框框线
    ax.spines['top'].set_visible(False)   
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 0.2)
    plt.ylim(0, 60) # 70
    # plt.yticks(np.arange(0, 71, 10))  # 设置y轴刻度，步进为10
    plt.yticks(np.arange(0, 60, 10))
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_histogram_range(values, title, xlabel, save_path, color):
    font_size, name_size = 18, 14
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    # plt.hist(
    #     values, bins=30, color=color, edgecolor='black', # 用30 bins表达
    #     weights=np.ones_like(values) / len(values) * 100
    # )
    bin_width = 0.02  # 设置 bin 宽度
    bins = np.arange(0, 1 + bin_width, bin_width)  # 按照宽度设置 bins
    plt.hist(values, bins=bins, color=color, edgecolor=None, #'black',
             weights=np.ones_like(values) / len(values) * 100)# 百分比
    ax.set_facecolor('#e6e9f0')  # 设置浅灰色背景

    # 去掉边框框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 设置标签和标题
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel('Frequency (%)', fontsize=font_size)
    plt.title(title, fontsize=font_size)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 30) # 50
    # plt.yticks(np.arange(0, 50, 10))  # 设置y轴刻度，步进为10
    plt.yticks(np.arange(0, 30, 5))
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_r_and_range(r_est_naive, r_est_second, r_gt, bound_naive, bound_second, save_path='plot.png', sigma="",
                     name_list=[]):
    # xtick_labels
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in
                name_list]  # "..."->"BraTS2021_00753.nii.gz"-> "00753"

    x = np.arange(len(r_gt))  # 根据 r_gt 的长度生成 x
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = 0.1
    # overall-bar: r_naive, r_second
    r_est_array = np.full_like(x, r_est_naive, dtype=float)  # extend dim
    ax.errorbar(
        x - dist,
        r_est_array,  # extend
        yerr=[r_est_array - bound_naive[:, 0], bound_naive[:, 1] - r_est_array],
        fmt='o',
        color='green',
        capsize=4,
        capthick=2,
        linewidth=2.5,
        zorder=3,
        label=r'$r$'
    )
    r_est_array = np.full_like(x, r_est_second, dtype=float)  # extend dim
    ax.errorbar(
        x + dist,
        r_est_array,  # extend
        yerr=[r_est_array - bound_second[:, 0], bound_second[:, 1] - r_est_array],
        fmt='o',
        color='blue',
        capsize=4,
        capthick=2,
        linewidth=2.5,
        zorder=3,
        label=r'$r_{corr,2}$'
    )

    # 绘制真实值（红色散点）
    ax.scatter(
        x,
        r_gt,
        color='red',
        label=r'$r_{gt}$',
        zorder=4,
        s=50
    )

    font_size, name_size = 18, 14
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='center', fontsize=name_size)
    ax.set_facecolor('#e6e9f0') 
    ax.set_xlabel('Volume', fontsize=font_size)
    ax.set_ylabel('Ratio', fontsize=font_size)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=font_size)  # outside

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f'Ratio and Confidence Interval(±{sigma}$\\sigma$)', fontsize=font_size)

    # 分别设置 x/y 轴刻度字体大小
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ce_and_range(r_est, r_gt, bound_ce, bound, save_path='plot.png', sigma="", name_list=[]):
    # xtick_labels
    case_ids = [os.path.basename(name).split('_')[-1].split('.')[0] for name in
                name_list]  # "..."->"BraTS2021_00753.nii.gz"-> "00753"
    x = np.arange(len(r_gt))  # 根据 r_gt 的长度生成 x
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = 0.1
    # 扩展 r_est 为数组（所有点相同）
    r_est_array = np.full_like(x, r_est, dtype=float)
    # overall-bar（用 bound）
    ax.errorbar(
        x - dist,
        r_est_array,  # 使用扩展后的数组
        yerr=[r_est_array - bound[:, 0], bound[:, 1] - r_est_array],
        fmt='o',
        color='green',
        capsize=4,
        capthick=2,
        linewidth=2.5,
        zorder=3,
        label=r'$r$'
    )
    # ce-区间块（用 bound_ce）
    ax.vlines(
        x - dist,
        bound_ce[:, 0],
        bound_ce[:, 1],
        color='lightgreen',
        linewidth=10,
        alpha=0.5,
        zorder=1,
        label=r'$I_{miscalib.}$'
    )
    # 绘制真实值（红色散点）
    ax.scatter(
        x,
        r_gt,
        color='red',
        label=r'$r_{gt}$',
        zorder=4,
        s=50
    )
    font_size, name_size = 18, 14
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='center', fontsize=name_size)
    ax.set_facecolor('#e6e9f0')  
    ax.set_xlabel('Volume', fontsize=font_size)
    ax.set_ylabel('Ratio', fontsize=font_size)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f'Ratio and Confidence Interval(±{sigma}$\\sigma$)', fontsize=font_size)

    # 分别控制 x 和 y 轴的刻度标签字体大小
    ax.tick_params(axis='x', labelsize=name_size)
    ax.tick_params(axis='y', labelsize=name_size)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
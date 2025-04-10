import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
import os
from nnunetv2.evaluation.plot_utils import plot_histogram, plot_r_and_range, plot_ce_and_range


# 读取 JSON 文件
root = 'nnUNet_results_inference/Brats2021_holdin/Dataset137_BraTS2021/2d_CE_DC/fold_0/ratio_metrics_prob/'
json_path = os.path.join(root, 'plot_binaryCE_sigma_for_CrEDC_1e4.json')

with open(json_path, 'r') as f:data = json.load(f)

# 提取数据
sigma=""
r_gt = np.array(data['r_gt']['r_gt'])
r_est = np.array(data['r_naive']['r_est'])
range_ce = np.array(data['r_naive']['range__ce+1std'])
bias = np.array(data['r_naive']['bias_r'])
# plot_histogram
plot_histogram(bias, title=f'Histogram of Ratio Bias (±{sigma}$\\sigma$)',
               xlabel='Ratio Bias', save_path='bias_hist.png', color='skyblue')
plot_histogram(range_ce, title=f'Histogram of Confidence Interval (±{sigma}$\\sigma$)',
               xlabel=f'Interval Length', save_path='range_hist.png', color='#A1D6B5')
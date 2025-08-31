import matplotlib.pyplot as plt
import numpy as np

models = [r"nnUNet$_{\mathrm{2d}}$", r"nnUNet$_{\mathrm{3d}}$", "nnFormer", "UNETR++"]

## line-strokes
ece = [94.23, 93.61, 87.95, 89.59]
v_bias = [93.61, 86.60, 81.92, 76.44]
CP = [71.34, 67.01, 67.40, 65.75]
C = [68, 68, 68, 68]

angles = np.linspace(0, 2*np.pi, len(models), endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])
ece_closed = ece + [ece[0]]
v_bias_closed = v_bias + [v_bias[0]]
CP_closed = CP + [CP[0]]
C_closed = C + [C[0]]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

# 设置极坐标属性
ax.set_theta_offset(np.pi / 2)     # 让第一个标签在上面
ax.set_theta_direction(-1)         # 顺时针方向
ax.set_ylim(60, 100)
# r_ticks = [60, 70, 80, 90, 100]
# ax.set_rticks(r_ticks)
# ax.set_yticklabels([str(tick) for tick in r_ticks], fontsize=14)  # 增大刻度字号


ax.set_rgrids([])  # 移除径向网格线
ax.spines['polar'].set_visible(False)  # 移除极坐标边框
ax.set_xticks(angles)
ax.set_xticklabels(models, fontsize=16)
ax.tick_params(axis='x', pad=10)
linewidth = 1.6 # 3 # 1.6
# 1
ax.plot(angles_closed, ece_closed, linewidth=linewidth, color="blue", label="Ours (ECE)")
ax.fill(angles_closed, ece_closed, alpha=0.1, color="blue")

# 2
ax.plot(angles_closed, v_bias_closed, linewidth=linewidth, color="green", label="Ours (V-Bias)")
ax.fill(angles_closed, v_bias_closed, alpha=0.1, color="green")

# 3
ax.plot(angles_closed, CP_closed, linewidth=linewidth, color="gray", label="CP")
ax.fill(angles_closed, CP_closed, alpha=0.3, color="gray")

# 4
ax.plot(angles_closed, C_closed, linewidth=linewidth, linestyle="dashed", color="red", label="C")
ax.fill(angles_closed, C_closed, alpha=0.1, color='red')

# 加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14)
plt.tight_layout()
plt.savefig("radar_plot_msd.png", dpi=300, bbox_inches="tight")
plt.close()
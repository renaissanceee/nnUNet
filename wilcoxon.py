from scipy.stats import wilcoxon
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json
import numpy as np

# before = [140, 130, 150, ...]
# after = [135, 130, 145, ...]

root = 'nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test/ratio_metrics_prob_ntr'

## ours
ours_json = load_json(join(root, 'bins15_ratio_min_width_0.68.json'))
ours_range = np.array([per_case["ratio"]["r_naive"]["range__ce+1std"] for per_case in ours_json["ratio_per_case"]])
## CP
cp_json = load_json(join(root, 'CP68_ratio.json'))
cp_range = np.array(cp_json["interval_per_case"]) # (251,2)
cp_range = cp_range[:,1]-cp_range[:,0]

stat, p = wilcoxon(ours_range, cp_range)
print(f"stats={stat}, p={p}") # stats=2371.0, p=1.7152605993815106e-31

## ours
ours_json = load_json(join(root, 'bins15_ratio_min_width_0.8.json'))
ours_range = np.array([per_case["ratio"]["r_naive"]["range__ce+1std"] for per_case in ours_json["ratio_per_case"]])
## CP
cp_json = load_json(join(root, 'CP80_ratio.json'))
cp_range = np.array(cp_json["interval_per_case"]) # (251,2)
cp_range = cp_range[:,1]-cp_range[:,0]

stat, p = wilcoxon(ours_range, cp_range)
print(f"stats={stat}, p={p}") # stats=2371.0, p=1.7152605993815106e-31

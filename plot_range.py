from nnunetv2.evaluation.naive_estimator import plot_r_and_range_dataset,plot_ce_and_range_dataset




## load json
paired_samples = "plot.json"
folder_pred = "./nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/"
# "val_interval": r, r_corr
plot_r_and_range_dataset(paired_samples, folder_pred, sigma="",step_size = 10)
plot_r_and_range_dataset(paired_samples, folder_pred, sigma=3,step_size = 10)
# "val_interval_sep": ce, ce+sigma
plot_ce_and_range_dataset(paired_samples, folder_pred, sigma="",step_size = 10)
plot_ce_and_range_dataset(paired_samples, folder_pred, sigma=3,step_size = 10)
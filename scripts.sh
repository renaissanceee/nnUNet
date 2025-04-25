# nnUNetv2_train->nnUNetv2_TS->nnUNetv2_predict
# ->nnUNetv2_ratio_estimator, nnUNetv2_evaluate_folder

# ------------- train -------------
# --TS lbfgs
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerDiceLoss
nnUNetv2_train 137 2d 0
## only TS
nnUNetv2_TS -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d -f 0 --TS lbfgs_50
# ------------- prob (.nii.gz) -------------
## --TS
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0 --TS lbfgs_50
# ------------- ratio -------------
## --ce_type {bins15,kde1,kde2,nll,bs}
## --TS {lbfgs_50, list_1000}, --binary
# before-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test --biomarker ntr --ce_type bins15
# after-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test --biomarker ntr --ce_type bins15 --TS lbfgs_50

# downsample for ratio
nnUNetv2_ratio_estimator_downsample /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test --biomarker ntr --ce_type bins15

# ------------- seg_metrics -------------
nnUNetv2_evaluate_folder  /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test --TS lbfgs_50









## for val-set
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsVal/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation --ce_type bins15 --TS lbfgs_50

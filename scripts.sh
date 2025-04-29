# nnUNetv2_train->nnUNetv2_TS->nnUNetv2_predict
# ->nnUNetv2_ratio_estimator(r), nnUNetv2_evaluate_folder(Dice)

# ------------- train -------------
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerDiceLoss --TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerTopk10Loss --TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerDiceTopK10Loss --TS
nnUNetv2_train 137 2d 0 --TS
## --TS {lbfgs_50, list_1000}
nnUNetv2_TS -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d -f 0 --TS lbfgs_50
# ------------- Val: get ECE(epsilon) -------------
## val(bins15):pred+ratio
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation_ece -d 137 -c 2d --save_probabilities -f 0
nnUNetv2_ratio_estimator_kde
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsVal_ece/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation_ece --biomarker ntr --ce_type bins15
## test(bins15)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0
nnUNetv2_ratio_estimator_kde
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test --biomarker ntr --ce_type bins15
## --TS {lbfgs_50, list_1000}

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

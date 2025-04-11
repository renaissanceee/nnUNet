

# ------------- train -------------
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS # t6 (doing)
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerDiceLoss --TS
nnUNetv2_train 137 2d 0 --TS

# ------------- prob (.nii.gz) -------------
# validation (--TS)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0 --TS
# validation_wo_TS (w/o)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0

# ------------- ratio -------------
# before-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json
# after-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --TS

# ------------- binary ratio -------------
# binary_ratio (before)
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --binary
# after
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --TS --binary


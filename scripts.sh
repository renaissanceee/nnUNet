# nnUNetv2_train->nnUNetv2_TS->nnUNetv2_predict
# ->nnUNetv2_ratio_estimator, nnUNetv2_evaluate_folder

# ------------- train -------------
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS lbfgs
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerDiceLoss --TS lbfgs
nnUNetv2_train 137 2d 0 --TS lbfgs

## only TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS adam -pretrained_weights nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/checkpoint_final.pth
## only TS
nnUNetv2_TS -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d -f 0 --TS lbfgs
# ------------- prob (.nii.gz) -------------
# test (--TS)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0 --TS adam
# test_wo_TS (w/o)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0
## for val-set
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0 
# ------------- ratio -------------
## --ce_type {bins,kde,nll,bs}
## --TS, --binary
# before-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json --ce_type bins
# after-TS
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json --ce_type bins --TS adam

## for val-set
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsVal/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json --ce_type bins --TS adam

# ------------- seg_metrics -------------
nnUNetv2_evaluate_folder  /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json --TS adam


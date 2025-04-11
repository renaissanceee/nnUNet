# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)


## Old Pipeline
```
source ~/.bashrc
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
```
train → inference(CE, Ratio) → TS → inference(CE, Ratio)
```
nnUNetv2_train → nnUNetv2_predict → nnUNetv2_ratio_estimator
e.g. CELoss/DiceLoss/2d_CE_DC
------------------ 1) Pretrain (100 epochs) ------------------
# train
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS
# infer (for prob)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0 -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0
# infer (for r)
optional. --binary (seg is used for r)
nnUNetv2_ratio_estimator  /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results_inference/Brats2021/Dataset137_BraTS2021/2d_CE/fold_0/ -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --binary
```
```
------------------ 2) .bashrc: → holdout ------------------
modify nnUNet_Trainer.py (xx_TS)
------------------ 3) TS (5 epochs) ------------------
nnUNetv2_train 137 2d all -tr nnUNetTrainerCELoss -pretrained_weights nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/checkpoint_final.pth
------------------ 4) .bashrc: → holdin ------------------
------------------ 5) inference ------------------
# infer (for prob)  --TS 
actually in predict_from_raw_data.py
nnUNetv2_predict -i nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results_inference/Brats2021_holdout/Dataset137_BraTS2021/2d_CE/fold_0 -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0 --TS 
# infer (for r)  --binary 
nnUNetv2_ratio_estimator  /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_holdin/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet/nnUNet_results_inference/Brats2021_holdout/Dataset137_BraTS2021/2d_CE/fold_0/ -pfile nnUNet_results/Brats2021_holdout/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021_holdout/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o binaryCE_sigma_for_CE_1e4.json --binary
```
```
choices:
--TS (load temperature.json)
--binary (seg is used for r)
```

## Path
```
# bashrc
/user/leuven/372/vsc37255/
# holdin
nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/
  |-temperature.json
  |-validation
    |-.nii.gz
    |-.npz
    |-.pkl
  |-checkpoint_final.pth
nnUNet_results_inference/Brats2021_holdin/Dataset137_BraTS2021/2d_CE/fold_0/
  |-.nii.gz(gt) .npz(prob) .pkl
  |-ratio_metrics_binary  (--binary)
  |-ratio_metrics_prob
     |-bins_interval_10_1e4
        |-binaryCE_sigma
           |-bias_hist_100.png
           |-range_hist_100.png
     |-plots_interval_10_1e4
        |-
           |-        
     |-binaryCE_sigma_for_CrE_1e4.json
     |-plot_binaryCE_sigma_for_CrE_1e4.json

# holdout

```
## Pipeline
```
source ~/.bashrc
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
nnUNetv2_train → nnUNetv2_predict → nnUNetv2_ratio_estimator
```
train → inference(CE, Ratio) → TS → inference(CE, Ratio)
```
e.g. CELoss/DiceLoss/2d_CE_DC
------------------ 1) Train → holdin(100)+holdout(5) ------------------
# train+TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS
# only TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss -pretrained_weights /staging/leuven/stg_00081/jli/calibration/nnUNet/nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/checkpoint_final.pth
------------------ 4) .bashrc: → holdin ------------------
------------------ 2) inference (prob) ------------------
--TS
# nnUNet_results/.../validation (--TS)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0 --TS 
# nnUNet_results/.../validation_wo_TS (w/o)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0
------------------ 3) inference (r) ------------------
--binary --TS
# validation/ratio_metrics_prob (--TS)
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --TS
# validation_wo_TS/ratio_metrics_binary (w/o TS, binary)
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/validation -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --binary
```
```
nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/
    |-validation
    |-validation_wo_TS
        |-ratio_metrics_prob
        |-ratio_metrics_binary
```
## Path
```
# bashrc
/user/leuven/372/vsc37255/
# holdin
nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/
  |-temperature.json
  |-validation
    |-.nii.gz
    |-.npz
    |-.pkl
  |-checkpoint_final.pth
nnUNet_results_inference/Brats2021_holdin/Dataset137_BraTS2021/2d_CE/fold_0/
  |-.nii.gz(gt) .npz(prob) .pkl
  |-ratio_metrics_binary  (--binary)
  |-ratio_metrics_prob
     |-bins_interval_10_1e4
        |-binaryCE_sigma
           |-bias_hist_100.png
           |-range_hist_100.png
     |-plots_interval_10_1e4
        |-
           |-        
     |-binaryCE_sigma_for_CrE_1e4.json
     |-plot_binaryCE_sigma_for_CrE_1e4.json

# holdout

```

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Learning from sparse annotations (scribbles, slices)](documentation/ignore_label.md)
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)


# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

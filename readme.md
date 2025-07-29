# nnUNet

## Pipeline
```
## BraTS21
python nnunetv2/dataset_conversion/Dataset137_BraTS21.py
source ~/.bashrc # -> change path
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
nnUNetv2_train → nnUNetv2_predict → nnUNetv2_ratio_estimator

## KiT23
python nnunetv2/dataset_conversion/Dataset220_KiTS2023.py /lustre1/project/stg_00081/jli/calibration/kits23/dataset
source ~/.bashrc
nnUNetv2_plan_and_preprocess -d 220 --verify_dataset_integrity
...
```
train → inference(CE, Ratio) → TS → inference(CE, Ratio)
```
e.g. CELoss/DiceLoss/2d_CE_DC
------------------ 1) Train → holdin(100)+holdout(5) ------------------
# train+TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss --TS
# only TS
nnUNetv2_train 137 2d 0 -tr nnUNetTrainerCELoss -pretrained_weights /staging/leuven/stg_00081/jli/calibration/nnUNet/nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/checkpoint_final.pth
------------------ 2) inference (prob) ------------------
--TS
# nnUNet_results/.../test_TS (--TS)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0 --TS 
# nnUNet_results/.../test (w/o)
nnUNetv2_predict -i nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/ -o nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -d 137 -c 2d --save_probabilities -f 0 --suffix fold_0
------------------ 3) inference (r) ------------------
--binary --TS --crop 190
# test_TS/ratio_metrics_prob (--TS)
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/test -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --TS
# test/ratio_metrics_binary (w/o TS, binary)
nnUNetv2_ratio_estimator /staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw_nested/Dataset137_BraTS2021/labelsTs/fold_0/  /staging/leuven/stg_00081/jli/calibration/nnUNet_nested/nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/tets -pfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/plans.json  -djfile nnUNet_results/Brats2021/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/dataset.json -o ratio_1e4.json --binary
```
```
nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/
    |-test_TS
    |-test
        |-ratio_metrics_prob
        |-ratio_metrics_binary
        |-ratio_metrics_prob_crop190&140 (optional)
```
## Path
```
# bashrc
/user/leuven/372/vsc37255/
# holdin
nnUNet_results/Brats2021_holdin/Dataset137_BraTS2021/nnUNetTrainerCELoss__nnUNetPlans__2d/fold_0/
  |-temperature.json
  |-test
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
---------------------
## Path_old
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

import numpy as np
import torch
from abstention.calibration import TempScaling
from abstention.label_shift import RLLSImbalanceAdapter
import torch.nn as nn

def get_importance_weights(valid_preds, valid_labels, shifted_test_preds):
    imbalance_adapter = RLLSImbalanceAdapter()# "rlls-hard"
    imbalance_adapter_func = imbalance_adapter(
        valid_labels=valid_labels,
        tofit_initial_posterior_probs=shifted_test_preds,
        valid_posterior_probs=valid_preds,
    )
    adapted_shifted_test_preds = imbalance_adapter_func(shifted_test_preds)

    return {
        "weights": imbalance_adapter_func.multipliers,
        "adapted_test_pred": adapted_shifted_test_preds,
    }
class EceLabelShift(nn.Module):
    """
    Compute ECE (Expected Calibration Error) under label shift
    """

    # def __init__(self, p=2, n_bins=15, adaptive_bins=True, classwise=False):
    def __init__(self, p=1, n_bins=15, adaptive_bins=True, classwise=True):
        super(EceLabelShift, self).__init__()
        self.p = p
        self.n_bins = n_bins
        self.adaptive_bins = adaptive_bins
        self.classwise = classwise

    def set_bins(self, confidences):
        if self.adaptive_bins:
            _, bin_boundaries = np.histogram(
                confidences.cpu().detach(),
                self.histedges_equalN(confidences.cpu().detach()),
            )
        else:
            bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.n_bins + 1), np.arange(npt), np.sort(x)
        )

    def get_ece(self, confidences_source, confidences_target, labels_source, weight):
        tmp_sum = torch.zeros(1, device=confidences_source.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin_source = confidences_source.gt(
                bin_lower.item()
            ) * confidences_source.le(bin_upper.item())
            in_bin_target = confidences_target.gt(
                bin_lower.item()
            ) * confidences_target.le(bin_upper.item())
            if in_bin_target.sum() > 1:
                normalizer = (len(confidences_target) - 1) / len(confidences_source)
                weighted_num = weight * labels_source[in_bin_source].float().sum()
                cond_expect = normalizer * weighted_num / (in_bin_target.sum() - 1)
                for point_target in confidences_target[in_bin_target]:
                    tmp_sum += torch.abs(point_target - cond_expect) ** self.p
        return tmp_sum / len(confidences_target)

    def get_ece_top_label(
        self,
        confidences_source,
        confidences_target,
        labels_source,
        predictions_source,
        weights,
    ):
        tmp_sum = torch.zeros(1, device=confidences_source.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin_source = confidences_source.gt(
                bin_lower.item()
            ) * confidences_source.le(bin_upper.item())
            in_bin_target = confidences_target.gt(
                bin_lower.item()
            ) * confidences_target.le(bin_upper.item())

            labels_source_in_bin = labels_source[in_bin_source]
            predictions_source_in_bin = predictions_source[in_bin_source]
            accuracies_source_in_bin = predictions_source_in_bin.eq(
                labels_source_in_bin
            )
            if in_bin_target.sum() > 1:
                normalizer = (len(confidences_target) - 1) / len(confidences_source)
                weighted_num = 0
                for c in range(len(weights)):
                    labels_source_in_bin_per_class_idx = labels_source_in_bin == c
                    class_mask_in_bin_per_class = accuracies_source_in_bin[
                        labels_source_in_bin_per_class_idx
                    ]
                    weighted_num += (
                        weights[c] * class_mask_in_bin_per_class.float().sum()
                    )

                cond_expect = normalizer * weighted_num / (in_bin_target.sum() - 1)
                for point_target in confidences_target[in_bin_target]:
                    tmp_sum += torch.abs(point_target - cond_expect) ** self.p

        return tmp_sum / len(confidences_target)

    # def forward(self, preds_target, preds_source, labels_source, weights, **kwargs):
    def forward(self, preds_target, preds_source, labels_source, weights):
        ## per-volume
        self.set_bins(preds_target)
        per_volume_ece = self.get_ece(
            preds_source,
            preds_target,
            labels_source,
            weights,
        )
        return per_volume_ece


## ---------------------------------------------- ##
# output = get_importance_weights(preds_source, labels_source, preds_target)# prob [N,4]
# weights = torch.tensor(output["weights"])
# ### example (label-shift)
# estimate = EceLabelShift(
#     preds_target=preds_target,
#     preds_source=preds_source,
#     labels_source=labels_source,
#     weights=weights[2], # for necrosis
# )

               
### example (boot+label-shift)
# ece_label_shift_estimator = BootstrapMeanVarEstimator(
#     estimator=EceLabelShift(
#         adaptive_bins=True, n_bins=args.n_bins, p=args.p, classwise=True
#     ),
# ece_label_shift, variance_label_shift = ece_label_shift_estimator(
#     logits_source=source_logits,
#     labels_source=source_labels,
#     logits=target_logits,
#     weights=weights,
# )


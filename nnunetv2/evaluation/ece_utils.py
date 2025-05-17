import torch
from nnunetv2.evaluation.ece_kde import get_ece_kde
import torch.nn.functional as F
import os
import numpy as np
from nnunetv2.evaluation.ece_label_shift import get_importance_weights, EceLabelShift, compute_true_w


def fast_ece(y_true, y_pred, bins=10, device='cuda'):
    y_true, y_pred = y_true.to(device), y_pred.to(device)
    # Create bins
    bins = torch.linspace(0., 1. - 1. / bins, bins, device=device)

    # Digitize - find which bin each prediction belongs to
    binids = (torch.bucketize(y_pred, bins, right=True) - 1).view(-1)  # Ensure binids is 1D

    # Compute statistics per bin
    bin_sums = torch.bincount(binids, weights=y_pred, minlength=len(bins)).to(device)
    bin_true = torch.bincount(binids, weights=y_true, minlength=len(bins)).to(device)
    bin_total = torch.bincount(binids, minlength=len(bins)).to(device)

    # Only consider non-empty bins
    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])  # accuracy
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])  # confidence
    weights = bin_total[nonzero] / torch.sum(bin_total[nonzero])

    # Calculate ECE
    l1 = torch.abs(prob_true - prob_pred)
    ece = torch.sum(weights * l1)
    return ece


def ece_loss(preds, labels, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=preds.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = torch.max(preds, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=preds.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def ece_loss_binary(preds, labels, bins=15):
    """
    preds: Tensor of shape [N, 1], confidence score from softmax for a single class
    labels: Tensor of shape [N], binary labels: 1 if sample belongs to this class, else 0
    """
    preds = preds.squeeze(1)  # [N]
    labels = labels.long()  # [N]

    bin_boundaries = torch.linspace(0, 1, bins + 1, device=preds.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=preds.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (preds > bin_lower) & (preds <= bin_upper)
        # in_bin = preds.gt(bin_lower) * preds.le(bin_upper)  # Bool mask

        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            avg_confidence_in_bin = preds[in_bin].mean()
            accuracy_in_bin = labels[in_bin].float().mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def brier_score(probabilities, labels):
    return torch.mean((probabilities - labels) ** 2)


def lp_score(probabilities, labels, p=1):
    return torch.norm(probabilities - labels, p=p) / error.numel()


def l1_score(probabilities, labels):
    # return torch.mean(torch.abs(probabilities - labels))
    return torch.mean(probabilities - labels)


def get_ece_kde_sub(f, y, bandwidth, p, mc_type, device, sub=1e4):
    # random permute -> batch -> average
    idx = torch.randperm(f.shape[0])  # 例如 tensor([3, 1, 7, ..., 0])
    f_shuffled, y_shuffled = torch.clamp(f[idx, :], min=0, max=1), y[idx]
    batch_size = int(sub)  # enough
    batch_f = f_shuffled[:batch_size].to(device)
    batch_y = y_shuffled[:batch_size].to(device)
    batch_ratio = get_ece_kde(batch_f, batch_y, bandwidth, p, mc_type, device).to("cpu")
    return batch_ratio


def calc_ece_kde(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, p):
    # print(f"Analyzing ece_kde with l-{p}...")
    # 1d_kde
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    bandwidth = 0.02  # 0.001
    device = "cuda"
    sub = 1e4
    epsilon_y = get_ece_kde_sub(tensor_nec_prob_map, tensor_nec_gt_map, bandwidth, p,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs 2
    epsilon_x = get_ece_kde_sub(tensor_wt_prob_map.to(device), tensor_wt_gt_map.to(device), bandwidth, p,
                                mc_type='canonical', device=device, sub=sub)  # binary: 0 vs {1,2,3}
    return epsilon_y, epsilon_x


def calc_bs(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    # print(f"Analyzing Brier_score ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1), tensor_wt_prob_map.reshape(-1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1).to(torch.int64), tensor_wt_gt_map.reshape(
        -1).to(torch.int64)
    epsilon_y = brier_score(tensor_nec_prob_map, tensor_nec_gt_map)
    epsilon_x = brier_score(tensor_wt_prob_map, tensor_wt_gt_map)
    return epsilon_y, epsilon_x


def calc_v_bias(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    v_bias_y = l1_score(tensor_nec_prob_map.reshape(-1), tensor_nec_gt_map.reshape(-1).to(torch.int64))
    v_bias_x = l1_score(tensor_wt_prob_map.reshape(-1), tensor_wt_gt_map.reshape(-1).to(torch.int64))
    return v_bias_y.item(), v_bias_x.item()


def calc_nll(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map):
    # print(f"Analyzing NLL ...")
    tensor_wt_prob_map = torch.clamp(tensor_wt_prob_map, min=0, max=1)
    epsilon_y = F.binary_cross_entropy(tensor_nec_prob_map.reshape(-1), tensor_nec_gt_map.reshape(-1).double(),
                                       reduction='mean')
    epsilon_x = F.binary_cross_entropy(tensor_wt_prob_map.reshape(-1), tensor_wt_gt_map.reshape(-1).double(),
                                       reduction='mean')
    return epsilon_y, epsilon_x


def calc_ece_bins(tensor_nec_prob_map, tensor_wt_prob_map, tensor_nec_gt_map, tensor_wt_gt_map, bins):
    # print(f"Analyzing ece_bins ...")
    tensor_nec_prob_map, tensor_wt_prob_map = tensor_nec_prob_map.reshape(-1, 1), tensor_wt_prob_map.reshape(-1, 1)
    tensor_nec_gt_map, tensor_wt_gt_map = tensor_nec_gt_map.reshape(-1), tensor_wt_gt_map.reshape(
        -1)  # .to(torch.int64)
    epsilon_y = ece_loss_binary(tensor_nec_prob_map, tensor_nec_gt_map, bins=bins)
    epsilon_x = ece_loss_binary(tensor_wt_prob_map, tensor_wt_gt_map, bins=bins)
    return epsilon_y, epsilon_x


def detect_failure(paired_samples):
    """Detect failure cases: r_gt falls outside of CE + Nσ bounds."""
    r_gt = paired_samples['r_gt']['r_gt']
    case_ids = np.array([
        os.path.basename(name).split('_')[-1].split('.')[0]
        for name in paired_samples['reference_file']
    ])

    def out_of_bound(mask_key):
        bounds = paired_samples['r_naive'][mask_key]
        lower, upper = bounds[:, 0], bounds[:, 1]
        return (r_gt < lower) | (r_gt > upper)

    return case_ids[out_of_bound('bound__ce+1std')]


def detect_failure_all(paired_samples):
    """Detect failure cases: r_gt falls outside of CE + Nσ bounds."""
    r_gt = paired_samples['r_gt']['r_gt']
    case_ids = np.array([
        os.path.basename(name).split('_')[-1].split('.')[0]
        for name in paired_samples['reference_file']
    ])

    def out_of_bound(mask_key):
        bounds = paired_samples['r_naive'][mask_key]
        lower, upper = bounds[:, 0], bounds[:, 1]
        return (r_gt < lower) | (r_gt > upper)

    return case_ids[out_of_bound('bound__ce+1std')], \
        case_ids[out_of_bound('bound__ce+2std')], \
        case_ids[out_of_bound('bound__ce+3std')]

def complement_tuple(a):
    full_set = {0, 1, 2, 3}
    a_set = set(a)
    b_set = full_set - a_set
    return tuple(sorted(b_set))


def arrange_for_binary_weights(preds_source, tensor_seg_prob_map, labels_source_one_hot, interested_region):
    complement = complement_tuple(interested_region)

    preds_source_nec = torch.stack([
        preds_source[:, list(interested_region)].sum(dim=1),
        preds_source[:, list(complement)].sum(dim=1)
    ], dim=1)

    tensor_seg_prob_map_for_nec = torch.stack([
        tensor_seg_prob_map[:, list(interested_region)].sum(dim=1),
        tensor_seg_prob_map[:, list(complement)].sum(dim=1)
    ], dim=1)

    labels_source_one_hot_nec = torch.stack([
        labels_source_one_hot[:, list(interested_region)].any(dim=1),
        labels_source_one_hot[:, list(complement)].any(dim=1)
    ], dim=1)

    return preds_source_nec, tensor_seg_prob_map_for_nec, labels_source_one_hot_nec


def calc_label_shift(tensor_nec_prob_map, tensor_wt_prob_map, tensor_seg_prob_map, source_dict, tensor_seg_gt_map):
    estimator = EceLabelShift(adaptive_bins=True, n_bins=15, p=1)
    # source_dict = {'nec_prob_map_source': nec_prob_map_source, 'wt_prob_map_source': wt_prob_map_source,
    #                'nec_gt_map_source': nec_gt_map_source, 'wt_gt_map_source': wt_gt_map_source,
    #                'preds_source':preds_source, 'labels_source_one_hot':labels_source_one_hot,
    #                'interested_region':interested_region}

    ## source
    nec_prob_map_source, wt_prob_map_source = source_dict['nec_prob_map_source'], source_dict['wt_prob_map_source']
    nec_gt_map_source, wt_gt_map_source = source_dict['nec_gt_map_source'], source_dict['wt_gt_map_source']
    interested_region = source_dict['interested_region']
    labels_source_one_hot = source_dict['labels_source_one_hot']
    preds_source = source_dict['preds_source']
    ## weights
    ## menthod_1: add_up
    tensor_seg_prob_map = tensor_seg_prob_map.permute(1, 2, 3, 0).reshape(-1, 4)
    output = get_importance_weights(preds_source.numpy(), labels_source_one_hot.numpy(),
                                    tensor_seg_prob_map.numpy())  # [N,4]
    weights = torch.tensor(output["weights"])  ## print("four class weights: ", weights)
    tensor_seg_gt_map = tensor_seg_gt_map.squeeze(0).reshape(-1)
    nec_weight, wt_weight = weights[list(interested_region)].sum(dim=0), weights[list((1, 2, 3))].sum(dim=0)
    true_weight = compute_true_w(np.argmax(labels_source_one_hot.numpy(), axis=1), tensor_seg_gt_map.numpy(), n_class=4)
    nec_weight_true, wt_weight_true = true_weight[list(interested_region)].sum(dim=0), true_weight[list((1, 2, 3))].sum(
        dim=0)
    ## menthod_2: binarize
    # nec
    # tensor_seg_prob_map = tensor_seg_prob_map.permute(1, 2, 3, 0).reshape(-1, 4)
    # preds_source_nec, tensor_seg_prob_map_for_nec, labels_source_one_hot_nec = arrange_for_binary_weights(preds_source,
    #                                                                                                       tensor_seg_prob_map,
    #                                                                                                       labels_source_one_hot,
    #                                                                                                       interested_region)
    # output = get_importance_weights(preds_source_nec.numpy(), labels_source_one_hot_nec.numpy(), tensor_seg_prob_map_for_nec.numpy())  # [N,4]
    # nec_weight = torch.tensor(output["weights"])[0] # 0:nec, 1:non-nec
    # print("weights (nec vs non-nec): ", nec_weight, torch.tensor(output["weights"])[1])
    # # wt
    # preds_source_wt, tensor_seg_prob_map_for_wt, labels_source_one_hot_wt = arrange_for_binary_weights(preds_source,
    #                                                                                                       tensor_seg_prob_map,
    #                                                                                                       labels_source_one_hot,
    #                                                                                                       (1,2,3))
    # output = get_importance_weights(preds_source_wt.numpy(), labels_source_one_hot_wt.numpy(),
    #                                 tensor_seg_prob_map_for_wt.numpy())  # [N,4]
    # wt_weight = torch.tensor(output["weights"])[0]  # 0:nec, 1:non-nec
    # print("weights (wt vs non-wt): ", wt_weight, torch.tensor(output["weights"])[1])

    device = 'cuda'
    nec_ece_true = estimator(preds_target=tensor_nec_prob_map.reshape(-1).to(device),
                             preds_source=nec_prob_map_source.reshape(-1).to(device),
                             labels_source=nec_gt_map_source.reshape(-1).to(device), weights=nec_weight_true.to(device))
    wt_ece_true = estimator(preds_target=tensor_wt_prob_map.reshape(-1).to(device),
                            preds_source=wt_prob_map_source.reshape(-1).to(device),
                            labels_source=wt_gt_map_source.reshape(-1).to(device), weights=wt_weight_true.to(device))
    # nec_ece_est = estimator(preds_target=tensor_nec_prob_map.reshape(-1).to(device),
    #                         preds_source=nec_prob_map_source.reshape(-1).to(device),
    #                         labels_source=nec_gt_map_source.reshape(-1).to(device), weights=nec_weight.to(device))
    # wt_ece_est = estimator(preds_target=tensor_wt_prob_map.reshape(-1).to(device),
    #                        preds_source=wt_prob_map_source.reshape(-1).to(device),
    #                        labels_source=wt_gt_map_source.reshape(-1).to(device), weights=wt_weight.to(device))
    print("necrosis:", nec_ece_true)
    print("WT:", wt_ece_true)

    device = 'cuda'
    nec_ece = estimator(
        preds_target=tensor_nec_prob_map.reshape(-1).to(device),
        preds_source=nec_prob_map_source.reshape(-1).to(device),
        labels_source=nec_gt_map_source.reshape(-1).to(device),
        weights=nec_weight.to(device)  # for necrosis
    )
    wt_ece = estimator(
        preds_target=tensor_wt_prob_map.reshape(-1).to(device),
        preds_source=wt_prob_map_source.reshape(-1).to(device),
        labels_source=wt_gt_map_source.reshape(-1).to(device),
        weights=wt_weight.to(device)  # for wt
    )
    return nec_ece, wt_ece
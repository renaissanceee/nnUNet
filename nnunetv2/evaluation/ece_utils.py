import torch

def fast_ece(y_true, y_pred, n_bins=10, device='cuda'):
    # Create bins
    bins = torch.linspace(0., 1. - 1. / n_bins, n_bins, device=device)

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

def brier_score(probabilities, labels):
    return torch.mean((probabilities - labels) ** 2)

def mean_lp_dist(probabilities, labels, p = 1):
    error = probabilities - labels
    return torch.norm(error, p=p) / error.numel()

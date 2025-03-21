import numpy as np

def fast_ece(y_true, y_pred, n_bins=10):
    bins = np.linspace(0., 1. - 1./n_bins, n_bins)
    binids = np.digitize(y_pred, bins) - 1

    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0  # don't use empty bins
    prob_true = (bin_true[nonzero] / bin_total[nonzero])  # acc
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])  # conf
    weights = bin_total[nonzero] / np.sum(bin_total[nonzero])
    l1 = np.abs(prob_true-prob_pred)
    ece = np.sum(weights*l1)
    mce = l1.max()
    l1 = l1.sum()
    return {"acc": prob_true, "conf": prob_pred, "ECE": ece, "MCE": mce, "l1": l1}

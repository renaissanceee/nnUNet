import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
class JDTLoss(_Loss):
    def __init__(self,
                 weight=None,
                 ignore_index=None,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 smooth=1e-3,
                 threshold=0.01,
                 norm=1,
                 log_loss=False,
                 ):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.norm = norm
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.weight = weight
        self.apply_nonlin = softmax_helper_dim1 # torch.sigmoid


    def forward(self, logits, label, loss_mask=None):
        """
            logits (torch.Tensor): (B, C, D1, D2, ...).
            label (torch.Tensor):(B, C, D1, D2, ...).
        """
        batch_size, num_classes = logits.shape[:2]

        logits = logits.view(batch_size, num_classes, -1)
        prob = self.apply_nonlin(logits)

        label = label.view(batch_size, num_classes, -1)
        loss = self.forward_loss(prob, label, loss_mask)

        return loss


    def forward_loss(self, prob, label, loss_mask):
        # if loss_mask != None:
        #     prob = prob * loss_mask
        #     label = label * loss_mask
        # prob_card = torch.norm(prob, p=self.norm, dim=2)
        # label_card = torch.norm(label, p=self.norm, dim=2)
        # diff_card = torch.norm(prob - label, p=self.norm, dim=2)
        # tp = (prob_card + label_card - diff_card) / 2
        # fp = prob_card - tp
        # fn = label_card - tp
        axes = [0] + list(range(2, prob.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(prob, label, axes, loss_mask, False)

        batch_size, num_classes = prob.shape[:2]
        active_classes = self.compute_active_classes(label, num_classes, (0, 2)) # no-use-here
        return self.forward_loss_mIoUD(tp, fp, fn, active_classes)


    def compute_active_classes(self, label, shape, dim):
        mask = torch.ones(shape, dtype=torch.bool)
        active_classes = torch.zeros(shape, dtype=torch.bool, device=label.device)
        active_classes[mask] = 1
        return active_classes

    def forward_loss_mIoUD(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp)
        # tp = torch.sum(tp, dim=0)
        # fp = torch.sum(fp, dim=0)
        # fn = torch.sum(fn, dim=0)
        loss_mIoUD = 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_mIoUD = loss_mIoUD[active_classes]
        return torch.mean(loss_mIoUD)
    #
    #
    # def forward_loss_mIoUIC(self, tp, fp, fn, active_classes):
    #     if torch.sum(active_classes) < 1:
    #         return 0. * torch.sum(tp), 0. * torch.sum(tp)
    #
    #     tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
    #
    #     if self.log_loss:
    #         loss_matrix = -torch.log(tversky)
    #     else:
    #         loss_matrix = 1.0 - tversky
    #
    #     if self.gamma > 1:
    #         loss_matrix **= self.gamma
    #
    #     if self.class_weights != None:
    #         class_weights = self.class_weights.unsqueeze(0).expand_as(loss_matrix)
    #         loss_matrix *= class_weights
    #
    #     loss_matrix *= active_classes
    #     loss_mIoUI = self.reduce(loss_matrix, active_classes, 1)
    #     loss_mIoUC = self.reduce(loss_matrix, active_classes, 0)
    #
    #     return loss_mIoUI, loss_mIoUC
    #
    #
    # def reduce(self, loss_matrix, active_classes, dim):
    #     active_sum = torch.sum(active_classes, dim)
    #     active_dim = active_sum > 0
    #     loss = torch.sum(loss_matrix, dim)
    #     loss = loss[active_dim] / active_sum[active_dim]
    #     loss = torch.mean(loss)
    #
    #     return loss
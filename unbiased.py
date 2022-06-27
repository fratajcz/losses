import torch
import torch.nn.functional as F


def split_losses(preds: torch.Tensor, truth: torch.Tensor, magnification: int, eps=1e-32):
    """
    preds: logits ([-inf, inf]) of predictions
    truth: binary tensor containing ground truth 1 for positives and 0 for unknowns
    magnification: by how much to magnify positive losses AFTER class imbalance has been corrected
    eps (optinal): tiny value to avoid division by zero

    automatically corrects class imbalance
    """
    loss = F.binary_cross_entropy_with_logits
    positives = preds[truth.bool()]
    unknowns = preds[~truth.bool()]

    imbalance_factor = unknowns.shape[0] / (positives.shape[0] + eps)
    i_f = imbalance_factor * magnification

    positive_plus = loss(positives, torch.ones_like(positives))
    positive_minus = loss(positives, torch.zeros_like(positives))

    unknown_minus = loss(unknowns, torch.zeros_like(unknowns))

    return positive_plus * i_f, positive_minus * i_f, unknown_minus


def upu(preds: torch.Tensor, truth: torch.Tensor, magnification: int = 2):
    """
    preds: logits ([-inf, inf]) of predictions (do not use sigmoid layer before this loss)
    truth: binary tensor containing ground truth 1 for positives and 0 for unknowns
    magnification: by how much to magnify positive losses AFTER class imbalance has been corrected

    Equation 16 in https://arxiv.org/pdf/2103.04683.pdf
    """
    positive_plus, positive_minus, unknown_minus = split_losses(preds, truth, magnification)
    return positive_plus - positive_minus + unknown_minus


def nnpu(preds: torch.Tensor, truth: torch.Tensor, magnification: int = 2):
    """
    preds: logits ([-inf, inf]) of predictions (do not use sigmoid layer before this loss)
    truth: binary tensor containing ground truth 1 for positives and 0 for unknowns
    magnification: by how much to magnify positive losses AFTER class imbalance has been corrected

    From my tests, this does not penalize false-positive unknowns as much as upu does.
    Equation 17 in https://arxiv.org/pdf/2103.04683.pdf
    """
    positive_plus, positive_minus, unknown_minus = split_losses(preds, truth, magnification)
    return positive_plus + torch.max(torch.Tensor((0, (unknown_minus - positive_minus))))

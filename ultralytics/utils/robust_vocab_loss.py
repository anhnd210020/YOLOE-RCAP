"""Training-only region-wise robust aggregation of negative vocabulary BCE entries."""

import torch
import torch.nn.functional as F


def robust_vocab_classification_loss(pred_scores, target_scores, target_scores_sum, bce, tau):
    """Preserve soft-positive BCE and aggregate zero-target BCE per anchor for tau > 0.

    In the audited YOLOE segmentation path every classification entry has unit
    weight. For each anchor, B is therefore its number of zero-target classes,
    and logsumexp(tau * loss) - log(B) is the log moment under q0 = 1 / B.
    """
    positive = target_scores > 0
    negative = target_scores == 0

    # Use the same BCE module and target cast as the baseline for all positive entries.
    positive_bce = bce(pred_scores, target_scores.to(pred_scores.dtype)).masked_select(positive).sum()

    # Select rows with B > 0 before per-row normalization or log-mean-exp.
    b = negative.sum(dim=-1)
    has_negative = b > 0
    negative_scores = pred_scores[has_negative].float()
    negative_mask = negative[has_negative]
    active_b = b[has_negative].to(negative_scores.dtype)

    negative_bce = F.softplus(negative_scores)
    masked_bce = negative_bce.masked_fill(~negative_mask, -torch.inf)
    maximum = masked_bce.max(dim=-1).values

    # Centered log-mean-exp identity:
    # log(mean(exp(tau * loss))) / tau
    #   = maximum + log1p(mean(expm1(tau * (loss - maximum)))) / tau.
    # Centering prevents overflow; expm1/log1p retain precision for small tau.
    centered = (negative_bce - maximum.unsqueeze(-1)).masked_fill(~negative_mask, -torch.inf)
    excess = torch.expm1(tau * centered).masked_fill(~negative_mask, 0).sum(dim=-1) / active_b
    risk = maximum + torch.log1p(excess) / tau
    robust_negative = (active_b * risk).sum()

    return (positive_bce + robust_negative) / target_scores_sum

"""Classification objective for Class-Balanced Fixed-Budget Replay (CB-Replay)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_NAME = "class_balanced_fixed_budget_replay"
LOSS_VARIANT = "focal_current_ce_replay"


def classification_loss(logits, labels, old_class_indices=(), *, gamma=1.0):
    """Mean Focal loss on current examples and CE on retained old examples.

    Both groups use the full seen-class head. Each draw contributes once to the
    batch mean, preserving the replay sampler's requested exposure ratio.
    """
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, labels[:, None]).squeeze(1)
    focal = -((1.0 - target_log_probs.exp()) ** gamma) * target_log_probs
    old_indices = torch.as_tensor(list(old_class_indices), device=labels.device, dtype=torch.long)
    replay_mask = torch.isin(labels, old_indices)
    ce = -target_log_probs
    return torch.where(replay_mask, ce, focal).mean()

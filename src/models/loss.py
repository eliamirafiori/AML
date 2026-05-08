import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CITELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for supervised CE (Cross-Entropy)
        self.beta = beta  # Weight for tethering/distillation
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, student_feats, teacher_feats, labels):
        s_img, s_txt = student_feats
        t_img, t_txt = teacher_feats

        # Standard CLIP Contrastive Loss
        # (Simplified: Normalized dot product between image and text)
        s_img = s_img / s_img.norm(dim=-1, keepdim=True)
        s_txt = s_txt / s_txt.norm(dim=-1, keepdim=True)
        logits = s_img @ s_txt.t() * self.logit_scale.exp()

        ground_truth = torch.arange(len(logits), device=logits.device)
        loss_con = (
            F.cross_entropy(logits, ground_truth)
            + F.cross_entropy(logits.t(), ground_truth)
        ) / 2

        # Tethering (Distillation) Loss
        # Keeps student features close to pre-trained teacher features
        loss_dist = F.mse_loss(s_img, t_img) + F.mse_loss(s_txt, t_txt)

        # Discriminative CE Loss (using the labels for few-shot)
        # Using images to predict class labels via text-projection
        loss_ce = F.cross_entropy(logits, labels)

        return loss_con + (self.alpha * loss_ce) + (self.beta * loss_dist)

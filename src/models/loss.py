import torch
import torch.nn as nn
import torch.nn.functional as F


class CITELoss(nn.Module):
    def __init__(self, lambd=1.0, eta=1.0):
        super().__init__()
        self.lambd = lambd  # Weight for Supervised Contrastive (Eq. 6)
        self.eta = eta  # Weight for Similarity Distillation (Eq. 6)
        # logit_scale is usually part of the CLIP model itself,
        # but kept here if your architecture requires it.
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward(self, student_feats, teacher_feats, labels, all_class_features):
        """
        student_feats: (s_img, s_txt) from fine-tuned model
        teacher_feats: (t_img, t_txt) from original CLIP
        labels: Class indices for the current batch
        all_class_features: Text embeddings for ALL unique classes in the task
        """
        s_img, s_txt = student_feats
        t_img, t_txt = teacher_feats

        # Normalize features for cosine similarity
        s_img = s_img / s_img.norm(dim=-1, keepdim=True)
        s_txt = s_txt / s_txt.norm(dim=-1, keepdim=True)
        t_img = t_img / t_img.norm(dim=-1, keepdim=True)
        t_txt = t_txt / t_txt.norm(dim=-1, keepdim=True)

        scale = self.logit_scale.exp()

        # Discriminative Visual-Text Alignment (L_DVA) - Eq. 2 & 3
        # Compare batch images against all possible class text embeddings
        logits_dva = (s_img @ all_class_features.t()) * scale
        loss_dva = F.cross_entropy(logits_dva, labels)

        # Supervised Contrastive Learning (L_SCL) - Eq. 4
        # This aligns images and texts while accounting for same-class instances in batch
        logits_scl = (s_img @ s_txt.t()) * scale

        # Create mask: 1 for same class, 0 for different class
        # In a standard setup, this often defaults to an identity matrix
        ground_truth = torch.arange(len(labels), device=labels.device)
        loss_scl = (
            F.cross_entropy(logits_scl, ground_truth)
            + F.cross_entropy(logits_scl.t(), ground_truth)
        ) / 2

        # Vision-Language Similarity Distillation (L_VLD) - Eq. 5
        # KL Divergence between Student similarity and Teacher similarity
        with torch.no_grad():
            logits_teacher = (t_img @ t_txt.t()) * scale
            p_teacher = F.softmax(logits_teacher, dim=-1)

        p_student = F.log_softmax(logits_scl, dim=-1)
        loss_vld = F.kl_div(p_student, p_teacher, reduction="batchmean")

        # Final Objective Function - Eq. 6
        return loss_dva + (self.lambd * loss_scl) + (self.eta * loss_vld)

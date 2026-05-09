import clip
import torch
import torch.nn as nn


class CLIPCITE(nn.Module):
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        super().__init__()
        # The Student: All parameters set to requires_grad = True
        self.student, self.preprocess = clip.load(model_name, device=device)

        # The Teacher: Parameters frozen to preserve original knowledge
        self.teacher, _ = clip.load(model_name, device=device)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        # Extract features from both for distillation
        s_image_feat = self.student.encode_image(image)
        s_text_feat = self.student.encode_text(text)

        with torch.no_grad():
            t_image_feat = self.teacher.encode_image(image)
            t_text_feat = self.teacher.encode_text(text)

        return (s_image_feat, s_text_feat), (t_image_feat, t_text_feat)

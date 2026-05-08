import torch
import clip

from models.clip_cite import CLIPCITE
from models.loss import CITELoss
from data.build_datasets import get_few_shot_loader

# Hyperparams
LR = 1e-6
SHOTS = 16
EPOCHS = 10

model = CLIPCITE().cuda()
loader = get_few_shot_loader("./data/train", model.preprocess, n_shots=SHOTS)
criterion = CITELoss(alpha=0.5, beta=0.5)
optimizer = torch.optim.AdamW(model.student.parameters(), lr=LR, weight_decay=0.1)

# Text tokens for classes (e.g., "a photo of a cat")
class_names = [c.replace("_", " ") for c in loader.dataset.dataset.classes]
text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).cuda()

for epoch in range(EPOCHS):
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()

        # In CLIP-CITE, text features depend on labels in the batch
        batch_text = text_tokens[labels]

        s_feats, t_feats = model(images, batch_text)
        loss = criterion(s_feats, t_feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

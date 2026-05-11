import torch
import clip
import kagglehub
import os

from src.models.clip_cite import CLIPCITE
from src.models.loss import CITELoss
from src.data.build_datasets import build_train_test_partition, get_csv_few_shot_loader, get_few_shot_loader
from src.utils.utility import get_all_class_features
from src.utils.meters import AverageMeter
from src.utils.checkpoint import save_checkpoint
from src.evaluation import validate


LR = 1e-6
SHOTS = 5
EPOCHS = 20
best_acc = 0.0

model = CLIPCITE().cuda()

path = kagglehub.competition_download("aml-group-lab")
print("Path to competition files:", path)

train_csv = os.path.join(path, "train.csv")
images_path = os.path.join(path, "images")

train_loader, class_names = get_csv_few_shot_loader(
    train_csv, images_path, model.preprocess, n_shots=SHOTS
)

criterion = CITELoss(lambd=0.5,eta=0.5).cuda()
optimizer = torch.optim.AdamW(model.student.parameters(), lr=LR, weight_decay=0.1)

class_names = [c for c in train_loader.dataset.dataset.classes]
text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).cuda()

for epoch in range(EPOCHS):
    model.train()
    loss_meter = AverageMeter()

    # Pre-compute class prototypes for this epoch
    all_class_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).cuda()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        all_class_features = model.student.encode_text(all_class_tokens)
        all_class_features = all_class_features / all_class_features.norm(dim=-1, keepdim=True)

        # Select the specific text prompts for the classes present in this batch
        batch_text = all_class_features[labels]

        # Forward pass
        s_feats, t_feats = model(images, batch_text)
        loss = criterion(s_feats, t_feats, labels, all_class_features)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record loss
        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch [{epoch}] Training Loss: {loss_meter.avg:.4f}")

    # Use the validate function from test.py
    current_acc = validate(model, train_loader, class_names)
    print(f"Epoch [{epoch}] Val Accuracy: {current_acc:.2f}%")

    is_best = current_acc > best_acc
    best_acc = max(current_acc, best_acc)

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "state_dict": model.student.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        },
        is_best,
        filename=f"checkpoint_epoch_{epoch}.pth.tar",
    )

print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")

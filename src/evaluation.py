import torch
import clip

from src.utils.meters import AverageMeter


@torch.no_grad()
def validate(model, val_loader, class_names):
    model.eval()
    top1 = AverageMeter()

    # Pre-compute text features for all classes once
    text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).cuda()
    text_features = model.student.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for images, labels in val_loader:
        images, labels = images.cuda(), labels.cuda()

        # Image features
        image_features = model.student.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits
        logits = (100.0 * image_features @ text_features.t()).softmax(dim=-1)

        # Calculate accuracy
        acc1 = (logits.argmax(1) == labels).float().mean()
        top1.update(acc1.item(), images.size(0))

    return top1.avg

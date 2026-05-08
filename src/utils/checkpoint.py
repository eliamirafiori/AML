import torch


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        import shutil

        shutil.copyfile(filename, "model_best.pth.tar")


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.student.load_state_dict(checkpoint["state_dict"])
    return checkpoint["epoch"], checkpoint["best_acc"]

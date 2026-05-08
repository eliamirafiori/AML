import torch


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename) # state contains epoch, best_acc, state_dict, optimizer
    if is_best: # if the current model is the best one, copy it to a separate file
        import shutil

        shutil.copyfile(filename, "model_best.pth.tar")


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.student.load_state_dict(checkpoint["state_dict"]) # load only the student model's state_dict
    return checkpoint["epoch"], checkpoint["best_acc"]

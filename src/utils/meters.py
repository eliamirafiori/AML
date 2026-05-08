class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset() 

    def reset(self): # used at the beginning of each epoch to reset the meter
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # each time a batch is processed, we call this function to update the meter with the new value and the number of samples in the batch
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

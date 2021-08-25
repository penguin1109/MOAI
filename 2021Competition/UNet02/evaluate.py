import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice import multiclass_dice_coeff


def evaluate(net, dataloader, device, n_class):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    loop = tqdm(dataloader)

    # iterate over the validation set
    for batch_idx, (image, mask) in enumerate(loop):
        # move images and labels to correct device and type
        image = image.to(device = device, dtype = torch.float32)
        mask = mask.to(device=device, dtype = torch.long)
        mask = F.one_hot(mask, 3).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0).float()
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_class).permute(0, 3, 1, 2).float()

            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, :1, ...], mask[:, :1, ...], reduce_batch_first=False)

    net.train()
    return dice_score / num_val_batches
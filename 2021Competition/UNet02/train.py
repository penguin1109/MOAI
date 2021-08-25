import logging
import sys
from pathlib import Path
from utils.util import load_checkpoint, save_checkpoint

import torch
import albumentationas as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import MOAIDataset
from evaluate import evaluate
from unet import UNET
from utils.dice import dice_loss

def train_fn(model, loader, device, optimizer, scheduler, loss_fn, scaler, n_classes):
    loop = tqdm(loader)

    for batch_idx, (image, mask) in enumerate(loop):
        images = images.to(device = DEVICE)
        masks = mask.permute(0, 3, 1, 2).float().to(device = DEVICE, dtype = torch.long)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = loss_fn(predictions, masks)\
                +dice_loss(F.softmax(predictions, dim = 1).float(), F.one_hot(masks, n_classes).permute(0,3,1,2).float(),
                multi_class = True)
        
        # backward
        optimizer.zero_grad(set_to_none = True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())


def main():
    # create dataset
    train_transform = A.Compose(
        [
            A.Rotate(limit = 50, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.3),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    valid_transform = A.compose(
        [
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )
    #create loaders
    train_loader, valid_loader = get_loaders(
        TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BATCH_SIZE,
        train_transform, valid_transform, rate = RATE,
        num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY
    )

    # set optimizer, loss, lr scheduler, loss scaling
    model = UNET(3, 3).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr = LEARNING_RATE, weight_decay = 1e-8, momentum = 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 2)
    scaler = torch.cuda.amp.GradScaler(enabled = amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    if LOAD_MODEL:
        load_checkpoint(torch.load('/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints/UNET_checkpoint.pth.tar'), model)
        check_accuracy(valid_loader, model, DEVICE)

    #start training
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        print("Training for epoch %d", epoch)
        train_fn(model,train_loader, DEVICE, optimizer, scheduler, criterion, scaler)

        # save checkpoints
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        val_score = evaluate(model, valid_loader, DEVICE, n_classes = 3)
        scheduler.step(val_score)

        






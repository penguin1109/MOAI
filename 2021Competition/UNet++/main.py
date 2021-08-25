import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
def validate(config, valid_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def main():
    model = UNetPP(ch_in = 1, ch_out = 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2)



    dcom_dirs = np.array(sorted(glob(image_dir + '/*/*')))
    mask_dirs = np.array(sorted(glob(mask_dir + '/*/*')))

    dcom_dirs, mask_dirs = train_test_split(dcom_dirs, test_size = 0.2)

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast()], p = 1),
        transforms.Resize(512, 512),
        transforms.Normalize(mean = [0.0], std = [1.0], max_pixel_value = 1.0)])
    
    val_transform = Compose([
        transforms.Resize(512, 512),
        transforms.Normalize(mean = [0.0], std = [1.0], max_pixel_value = 1.0)
    ])

    train_loader, valid_loader = get_loaders(
        dcom_dirs, mask_dirs, BATCH_SIZE,
        train_transform, val_transform, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoints(torch.load('/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints/unetpp_checkpoint.pth.tar'), model)

    best_iou, trigger = 0, 0

    for epoch in range(NUM_EPOCHS):
        print("Training for epoch %d", epoch)

        train_log = train(config, train_loader, model, criterion, optimizer)
        valid_log = validate(config, valid_loader, model, criterion)

        scheduler.step(valid_log['loss'])

        trigger += 1

        if valid_log['iou' > best_iou:
            save_checkpoint()
            best_iou = valid_log['iou']
            print("=> SAVED BEST IOU")
            trigger = 0
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()




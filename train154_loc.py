import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from apex import amp

from adamw import AdamW
from losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeNet154_Unet_Loc

from imgaug import augmenters as iaa

from utils import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import gc

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

train_dirs = ['train', 'tier3']

models_folder = 'weights'

input_shape = (480, 480)


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(d, 'images', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        
        if random.random() > 0.96:
            img = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.6:
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]

        if random.random() > 0.1:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                msk0 = np.rot90(msk0, k=rot)

        if random.random() > 0.7:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            
        if random.random() > 0.4:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.2:
            crop_size = random.randint(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 5)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk0[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]

        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.95:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9597:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.92:
            if random.random() > 0.92:
                img = clahe(img)
            elif random.random() > 0.92:
                img = gauss_noise(img)
            elif random.random() > 0.92:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.92:
            if random.random() > 0.92:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.92:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.92:
                img = contrast(img, 0.9 + random.random() * 0.2)
                
        if random.random() > 0.95:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


    
class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


def validate(net, data_loader):
    dices0 = []

    _thr = 0.5

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            
            out = model(imgs)

            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
            
            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))

    d0 = np.mean(dices0)

    print("Val Dice: {}".format(d0))
    return d0


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score



def train_epoch(current_epoch, seg_loss, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        
        out = model(imgs)

        loss = seg_loss(out, msks)

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))

        dices.update(dice_sc, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))
        
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Dice {dice.avg:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    
    seed = int(sys.argv[1])
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    cudnn.benchmark = True

    batch_size = 14
    val_batch_size = 4

    snapshot_name = 'se154_loc_{}_1'.format(seed)

    train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)

    np.random.seed(seed + 321)
    random.seed(seed + 321)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    model = SeNet154_Unet_Loc().cuda()

    params = model.parameters()

    optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7, 11, 15, 19, 23, 27, 33, 41, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)

    model = nn.DataParallel(model).cuda()

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 14.0}, per_image=False).cuda() #True

    best_score = 0
    _cnt = -1
    torch.cuda.empty_cache()
    for epoch in range(30):
        train_epoch(epoch, seg_loss, model, optimizer, scheduler, train_data_loader)
        torch.cuda.empty_cache()
        best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
import os

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn

from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import SeResNext50_Unet_Loc, Dpn92_Unet_Loc, Res34_Unet_Loc, SeNet154_Unet_Loc

from utils import *

from sklearn.model_selection import train_test_split

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = 'test/images'
pred_folder = 'pred_loc_val'
train_dirs = ['train', 'tier3']
models_folder = 'weights'

all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(d, 'images', f))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(pred_folder, exist_ok=True)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    cudnn.benchmark = True

    models = []
    model_idxs = []

    tot_len = 0
    for seed in [0, 1, 2]:
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)
        tot_len += len(val_idxs)
        model_idxs.append(val_idxs)
        model_idxs.append(val_idxs)
        model_idxs.append(val_idxs)
        model_idxs.append(val_idxs)

        snap_to_load = 'res50_loc_{}_0_best'.format(seed)

        model = SeResNext50_Unet_Loc().cuda()

        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        model.eval()
        models.append(model)

        snap_to_load = 'dpn92_loc_{}_0_best'.format(seed)

        model = Dpn92_Unet_Loc().cuda()

        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        model.eval()
        models.append(model)

        snap_to_load = 'se154_loc_{}_0_best'.format(seed)

        model = SeNet154_Unet_Loc().cuda()
        
        model = nn.DataParallel(model).cuda()

        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        model.eval()
        models.append(model)

        snap_to_load = 'res34_loc_{}_1_best'.format(seed)

        model = Res34_Unet_Loc().cuda()
        
        model = nn.DataParallel(model).cuda()

        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
                .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        model.eval()
        models.append(model)

    
    unique_idxs = np.unique(np.asarray(model_idxs))

    print(tot_len, len(unique_idxs))

    with torch.no_grad():
        for idx in tqdm(unique_idxs):
            fn = all_files[idx]

            f = fn.split('/')[-1]

            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img = preprocess_inputs(img)

            inp = []
            inp.append(img)
            inp.append(img[::-1, ::-1, ...])
            inp = np.asarray(inp, dtype='float')
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp).cuda()

            pred = []
            _i = -1
            for model in models:
                _i += 1
                if idx not in model_idxs[_i]:
                    continue
                msk = model(inp)
                msk = torch.sigmoid(msk)
                msk = msk.cpu().numpy()
                
                pred.append(msk[0, ...])
                pred.append(msk[1, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0)
            
            msk = pred_full * 255
            msk = msk.astype('uint8').transpose(1, 2, 0)
            cv2.imwrite(path.join(pred_folder, '{0}.png'.format(f.replace('.png', '_part1.png'))), msk[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
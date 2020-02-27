import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
from os import path, makedirs
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import torch
torch.set_num_threads(1)
from torch import nn
from torch.autograd import Variable
import timeit
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = ''

import gc

from zoo.models import SeNet154_Unet_Loc

from utils import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

models_folder = 'weights' #/work/

if __name__ == '__main__':
    t0 = timeit.default_timer()

    pre_file = sys.argv[1]
    post_file = sys.argv[2]
    loc_pred_file = sys.argv[3]
    cls_pred_file = sys.argv[4]

    pred_folder = 'pred154_loc'
    makedirs(pred_folder, exist_ok=True)

    models = []

    for seed in [0, 1, 2]:
        snap_to_load = 'se154_loc_{}_1_best'.format(seed)
        model = SeNet154_Unet_Loc(pretrained=None)
        model = nn.DataParallel(model)
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

        del loaded_dict
        del sd
        del checkpoint

    gc.collect()

    with torch.no_grad():
        img = cv2.imread(pre_file, cv2.IMREAD_COLOR)
        img = preprocess_inputs(img)

        inp = []
        inp.append(img)
        inp.append(img[::-1, ...])
        inp.append(img[:, ::-1, ...])
        inp.append(img[::-1, ::-1, ...])
        inp = np.asarray(inp, dtype='float')
        inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
        inp = Variable(inp)

        pred = []
        
        for model in models:
            for j in range(4):
                msk = model(inp[j:j+1])
                msk = torch.sigmoid(msk)
                msk = msk.cpu().numpy()
                
                #for tta to not crash on memory
                if j == 0:
                    pred.append(msk[0, ...])
                elif j == 1:
                    pred.append(msk[0, :, ::-1, :])
                elif j == 2:
                    pred.append(msk[0, :, :, ::-1])
                elif j == 3:
                    pred.append(msk[0, :, ::-1, ::-1])

        pred_full = np.asarray(pred).mean(axis=0)
        
        msk = pred_full * 255
        msk = msk.astype('uint8').transpose(1, 2, 0)

        f = os.path.basename(loc_pred_file)
        cv2.imwrite(path.join(pred_folder, '{0}'.format(f + '_part1.png')), msk[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
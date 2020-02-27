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

from zoo.models import Res34_Unet_Double

from utils import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

models_folder = 'weights' #/work/

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])

    pre_file = sys.argv[2]
    post_file = sys.argv[3]
    loc_pred_file = sys.argv[4]
    cls_pred_file = sys.argv[5]

    pred_folder = 'res34cls2_{}_tuned'.format(seed)
    makedirs(pred_folder, exist_ok=True)

    models = []

    snap_to_load = 'res34_cls2_{}_tuned_best'.format(seed)

    model = Res34_Unet_Double(pretrained=None)

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

    with torch.no_grad():
        img = cv2.imread(pre_file, cv2.IMREAD_COLOR)
        img2 = cv2.imread(post_file, cv2.IMREAD_COLOR)

        img = np.concatenate([img, img2], axis=2)
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
            for j in range(2):
                msk = model(inp[j*2:j*2+2])
                msk = torch.sigmoid(msk)
                msk = msk.cpu().numpy()
                
                if j == 0:
                    pred.append(msk[0, ...])
                    pred.append(msk[1, :, ::-1, :])
                else:
                    pred.append(msk[0, :, :, ::-1])
                    pred.append(msk[1, :, ::-1, ::-1])

        pred_full = np.asarray(pred).mean(axis=0)
        
        msk = pred_full * 255
        msk = msk.astype('uint8').transpose(1, 2, 0)

        f = os.path.basename(cls_pred_file)
        cv2.imwrite(path.join(pred_folder, '{0}'.format(f + '_part1.png')), msk[..., :3], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(path.join(pred_folder, '{0}'.format(f + '_part2.png')), msk[..., 2:], [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
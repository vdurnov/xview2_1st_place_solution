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

import timeit
import cv2

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

pred_folders = ['dpn92cls_0_tuned', 'dpn92cls_1_tuned', 'dpn92cls_2_tuned', 'res34cls2_0_tuned', 'res34cls2_1_tuned', 'res34cls2_2_tuned', 
                'res50cls_0_tuned', 'res50cls_1_tuned', 'res50cls_2_tuned', 'se154cls_0_tuned', 'se154cls_1_tuned', 'se154cls_2_tuned']

loc_folders = ['pred50_loc_tuned', 'pred92_loc_tuned', 'pred34_loc', 'pred154_loc']

_thr = [0.38, 0.13, 0.14]


if __name__ == '__main__':
    t0 = timeit.default_timer()

    pre_file = sys.argv[1]
    post_file = sys.argv[2]
    loc_pred_file = sys.argv[3]
    cls_pred_file = sys.argv[4]

    loc_fn = os.path.basename(loc_pred_file)
    loc_fn = '{0}'.format(loc_fn + '_part1.png')
    cls_fn = os.path.basename(cls_pred_file)

    preds = []
    for d in pred_folders:
        msk1 = cv2.imread(path.join(d, '{0}'.format(cls_fn + '_part1.png')), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(d, '{0}'.format(cls_fn + '_part2.png')), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        preds.append(msk)
    preds = np.asarray(preds).astype('float').sum(axis=0) / len(pred_folders) / 255
    
    loc_preds = []
    for d in loc_folders:
        msk = cv2.imread(path.join(d, loc_fn), cv2.IMREAD_UNCHANGED)
        loc_preds.append(msk)
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / len(loc_folders) / 255

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')

    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(loc_pred_file, msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(cls_pred_file, msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
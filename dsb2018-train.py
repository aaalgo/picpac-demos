#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import math 
from glob import glob 
import cv2
import random
import copy 
import numpy as np
import imageio
from skimage import measure
import logging
import subprocess as sp

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')

class Sample:
    def __init__ (self, folder):
        path = None
        for p in glob(os.path.join(folder, "images/*")):
            assert ".png" in p
            assert path is None
            path = p
            pass
        image = cv2.imread(path, -1) #cv2.IMREAD_COLOR)
        if len(image.shape) == 2:
            # Sanity check
            # In case of gray image data in private phase
            # case this to fail remove the assertion
            assert False
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            assert len(image.shape) == 3
            if image.shape[2] == 4:
                assert (image[:, :, 3] == 255).all()
                image = image[:, :, :3]
                pass
        assert image.shape[2] == 3

        self.image = image

        mask_shape = image.shape[:2]
        masks = []
        for path in glob(os.path.join(folder, "masks/*")):
            assert ".png" in path
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert mask.shape == mask_shape
            mask = np.clip(mask, 0, 1)
            masks.append(mask)
        self.masks = masks
        pass

    def mask_sum (self, dilate = None):
        v = np.zeros(self.image.shape[:2], dtype=np.float)
        for mask in self.masks:
            if dilate is None:
                v += mask
            elif dilate > 0:
                v += cv2.dilate(mask, (dilate, dilate))
            elif dilate < 0:
                # !!! check
                v += cv2.erode(mask, (-dilate, -dilate))
        return v


try:
    os.remove('dsb2018.db')
except:
    pass
db = picpac.Writer('dsb2018.db')
logging.warn("Importing images into PicPac database...")
for folder in glob(os.path.join(DATA_DIR, "stage1_train/*")):
    print(folder)
    sample = Sample(folder)
    sample.stat()
    image_buffer = cv2.imencode('.png', sample.image)[1].tostring()

    mask = np.clip(sample.mask_sum(), 0, 1).astype(np.uint8)
    mask_buffer = cv2.imencode('.png', mask)[1].tostring()
    db.append(image_buffer, mask_buffer)
del db
#print(folder)

sp.check_call('./fcn-train.py --db dsb2018.db --model dsb2018.model --annotate image')




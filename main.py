# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

from lib.utils import getMask_onlineEvaluation, learningBasedMatting

fn_im = cv2.cvtColor(cv2.imread('input_lowres/donkey.png'), cv2.COLOR_BGR2RGB)
fn_mask = getMask_onlineEvaluation('trimap_lowres/Trimap1/donkey.png')

alpha = learningBasedMatting(fn_im, fn_mask)

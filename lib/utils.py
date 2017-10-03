# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def debug_mat(matt):
    scipy.io.savemat('/Users/whitelok/Desktop/LearningBasedMatting_Matlab/debug.mat', {'py_mat': matt})

def compLapCoeff(winI,lambda_):
    if lambda_ is None:
        lambda_ = 0.00001
    Xi = winI
    Xi = np.concatenate((Xi, np.ones((Xi.shape[0], 1))), axis=1)
    I = np.eye(Xi.shape[0])
    I[-1][-1] = 0

    fenmu = np.dot(Xi, Xi.T) + lambda_ * I
    F = np.dot(np.dot(Xi, np.transpose(Xi)), np.linalg.inv(fenmu))
    I_F = np.eye(F.shape[0]) - F
    return np.dot(np.transpose(I_F), I_F)

def getMask_onlineEvaluation(mask_img_path):
    img = cv2.imread(mask_img_path, 0)
    mask = np.zeros(img.shape)
    force = [img == 255]
    back = [img == 0]
    mask[force] = 1
    mask[back] = -1
    return mask

def getLap_iccv09_overlapping(imdata, winsz, mask, lambda_):
    if lambda_ is None:
        lambda_ = 0.00001

    if len(winsz) == 1:
        winsz.append(winsz[0])

    tmp_imdata = np.double(imdata) / 255

    debug_mat(tmp_imdata)

    imsz = imdata.shape
    d = imdata.shape[2]

    pixInds = np.reshape(np.array([i for i in range(1, imsz[0] * imsz[1] + 1)]), (imsz[0], imsz[1]))

    winsz = np.array(winsz)
    winsz[winsz % 2 == 0] = winsz[winsz % 2 == 0] + 1
    numPixInWindow = winsz[0] * winsz[1]
    halfwinsz = (winsz - 1) / 2

    tmp_mask = np.abs(mask)
    tmp_mask[tmp_mask != 0] = 1
    scribble_mask = tmp_mask
    scribble_mask = cv2.erode(scribble_mask, np.ones((max(winsz), max(winsz))))

    numPix4Training = np.sum(1 - scribble_mask[halfwinsz[0]:-halfwinsz[0], halfwinsz[1]:-halfwinsz[1]])
    numNonzeroValue = numPix4Training * (numPixInWindow ** 2)

    row_inds = np.zeros((numNonzeroValue, 1))
    col_inds = np.zeros((numNonzeroValue, 1))
    vals = np.zeros((numNonzeroValue, 1))

    tmp_winData = None
    length = 0

    for j in range(halfwinsz[1] + 1, imsz[1] - halfwinsz[1]):
        for i in range(halfwinsz[0] + 1, imsz[0] - halfwinsz[0]):
            if scribble_mask[i][j] == 1:
                continue
            # winData=imdata(i-halfwinsz(1):i+halfwinsz(1),j-halfwinsz(2):j+halfwinsz(2),:);
            # winData=reshape(winData,numPixInWindow,d);
            # lapcoeff=compLapCoeff(winData,lambda);
            # print (i - halfwinsz[0]), (i + halfwinsz[0] + 1), (j - halfwinsz[1]), (j + halfwinsz[1] + 1)
            winData = tmp_imdata[(i-halfwinsz[0]):(i+halfwinsz[0]+1), (j-halfwinsz[1]):(j+halfwinsz[1]+1), :]
            # tmp_winData = winData
            winData = np.reshape(winData, (numPixInWindow, d))
            lapcoeff = compLapCoeff(winData, lambda_);

            # win_inds=pixInds(i-halfwinsz(1):i+halfwinsz(1),j-halfwinsz(2):j+halfwinsz(2));
            # row_inds(1+len:numPixInWindow^2+len)=reshape(repmat(win_inds(:),1,numPixInWindow),numPixInWindow^2,1);
            # col_inds(1+len:numPixInWindow^2+len)=reshape(repmat(win_inds(:)',numPixInWindow,1),numPixInWindow^2,1);
            # vals(1+len:numPixInWindow^2+len)=lapcoeff(:);
            # len=len+numPixInWindow^2;

            win_inds = pixInds[(i-halfwinsz[0]):(i+halfwinsz[0]+1), (j-halfwinsz[1]):(j+halfwinsz[1]+1)]
            print i, j
            print win_inds
            break
            # row_inds[length:(numPixInWindow ** 2+length)]=np.reshape(repmat(win_inds(:),1,numPixInWindow),numPixInWindow^2,1)
            # col_inds[length:(numPixInWindow ** 2+length)]=np.reshape(repmat(win_inds(:),numPixInWindow,1),numPixInWindow^2,1)
    # print winData.shape
    # print winData

def getLap(imdata, winsz, mask, lambda_):
    print 'Computing Laplacian matrix ... ...'
    return getLap_iccv09_overlapping(imdata, winsz, mask, lambda_)

def learningBasedMatting(im_data, mask):
    winsz = [3]
    c = 800
    lambda_ = 0.0000001

    L = getLap(im_data,winsz,mask,lambda_)

    # C = getC(mask, c)
    #
    # alpha_star = getAlpha_star(mask)
    #
    # alpha = solveQurdOpt(L, C, alpha_star)

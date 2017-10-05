# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy.matlib
import scipy.sparse.linalg

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix, spdiags, eye

def solveQurdOpt(L, C, alpha_star):
    print 'Solving quadratic optimization problem ... ...'
    lambda_ = 1e-6
    D = eye(L.shape[0], L.shape[1])
    tmp_mat_1 = np.reshape(alpha_star, (alpha_star.shape[0] * alpha_star.shape[1], 1), order='F').copy()
    tmp_mat_1 = C * tmp_mat_1
    # alpha = (L + C + D * lambda_) * np.linalg.inv(tmp_mat_1)
    print (L + C + D * lambda_).shape
    alpha = (tmp_mat_1) * scipy.sparse.linalg.inv(L + C + D * lambda_)

    alpha = np.reshape(alpha, (alpha_star.shape[0], alpha_star.shape[1]), order='F')
    tmp_mat_2 = np.squeeze(np.asarray(np.reshape(alpha_star, (alpha_star.shape[0] * alpha_star.shape[1], 1), order='F'))).copy().min()
    if np.squeeze(np.asarray(np.reshape(alpha_star, (alpha_star.shape[0] * alpha_star.shape[1], 1), order='F'))).min() == -1:
        alpha = alpha * 0.5 + 0.5;
    # alpha = max(min(alpha,1),0)

def getAlpha_star(mask):
    print 'Computing preknown alpha values ... ...'

    alpha_star = np.zeros((mask.shape[0], mask.shape[1]));
    alpha_star[mask > 0] =1
    alpha_star[mask > 0] = -1
    return alpha_star

def getC(mask, c):
    print 'Computing regularization matrix ... ...'
    # scribble_mask=abs(mask)~=0;
    tmp_mask = np.abs(mask).copy()
    tmp_mask[tmp_mask != 0] = 1
    scribble_mask = tmp_mask.copy()

    # numPix=size(mask,1)*size(mask,2);
    numPix = mask.shape[0] * mask.shape[1]

    # C=c*spdiags(double(scribble_mask(:)),0,numPix,numPix);
    tmp_m = np.squeeze(np.asarray(np.double(np.reshape(scribble_mask, (scribble_mask.shape[0] * scribble_mask.shape[1], 1), order='F'))))
    return c * spdiags(tmp_m, 0, numPix, numPix)

def debug_mat(name, matt):
    name = 'py_' + name
    scipy.io.savemat('/Users/whitelok/Desktop/LearningBasedMatting_Matlab/%s.mat' % name, {name: matt})

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

    imdata = np.double(imdata) / 255

    imsz = imdata.shape
    d = imdata.shape[2]

    pixInds = np.reshape(np.array([i for i in range(1, imsz[0] * imsz[1] + 1)]), (imsz[0], imsz[1]), order="F")

    winsz = np.array(winsz)
    winsz[winsz % 2 == 0] = winsz[winsz % 2 == 0] + 1
    numPixInWindow = winsz[0] * winsz[1]
    halfwinsz = (winsz - 1) / 2

    tmp_mask = np.abs(mask).copy()
    tmp_mask[tmp_mask != 0] = 1
    scribble_mask = tmp_mask.copy()
    scribble_mask = cv2.erode(scribble_mask, np.ones((max(winsz), max(winsz))))

    numPix4Training = np.sum(1 - scribble_mask[halfwinsz[0]:-halfwinsz[0], halfwinsz[1]:-halfwinsz[1]])
    numNonzeroValue = numPix4Training * (numPixInWindow ** 2)

    row_inds = np.zeros((numNonzeroValue, 1))
    col_inds = np.zeros((numNonzeroValue, 1))
    vals = np.zeros((numNonzeroValue, 1))

    tmp_winData = None
    length = 0

    axis_record = []

    for j in range(halfwinsz[1], (imsz[1] - halfwinsz[1])):
        for i in range(halfwinsz[0], (imsz[0] - halfwinsz[0])):
            if scribble_mask[i][j] == 1:
                continue

            winData = imdata[(i-halfwinsz[0]):(i+halfwinsz[0]+1), (j-halfwinsz[1]):(j+halfwinsz[1]+1), :]
            winData = np.reshape(winData, (numPixInWindow, d), order="F").copy()
            lapcoeff = compLapCoeff(winData, lambda_)
            win_inds = pixInds[(i-halfwinsz[0]):(i+halfwinsz[0]+1), (j-halfwinsz[1]):(j+halfwinsz[1]+1)]
            row_inds[length:(numPixInWindow ** 2 + length)]=np.reshape(np.matlib.repmat(win_inds[:], 1, numPixInWindow),(numPixInWindow ** 2, 1), order="F").copy()
            col_inds[length:(numPixInWindow ** 2 + length)]=np.reshape(np.matlib.repmat(np.transpose(win_inds[:]), numPixInWindow, 1),(numPixInWindow ** 2, 1), order="F").copy()
            vals[length:(numPixInWindow ** 2 + length)] = np.reshape(lapcoeff[:], (lapcoeff.shape[0] * lapcoeff.shape[1], 1)).copy()
            length = length + numPixInWindow ** 2

    # print row_inds[0], row_inds[2380832], row_inds[4761665], np.rank(row_inds), row_inds.shape
    # print col_inds[0], col_inds[2380832], col_inds[4761665], np.rank(col_inds), col_inds.shape
    # print vals[0], vals[2380832], vals[4761665], np.rank(vals), vals.shape

    row_inds = np.squeeze(np.asarray(row_inds))
    col_inds = np.squeeze(np.asarray(col_inds))
    vals = np.squeeze(np.asarray(vals))

    return coo_matrix((vals, (row_inds, col_inds)), shape=(imsz[0]*imsz[1], imsz[0]*imsz[1])).tocsr()

def getLap(imdata, winsz, mask, lambda_):
    print 'Computing Laplacian matrix ... ...'
    return getLap_iccv09_overlapping(imdata, winsz, mask, lambda_)

def learningBasedMatting(im_data, mask):
    winsz = [3]
    c = 800
    lambda_ = 0.0000001

    L = getLap(im_data,winsz,mask,lambda_)

    C = getC(mask, c)

    alpha_star = getAlpha_star(mask)

    alpha = solveQurdOpt(L, C, alpha_star)

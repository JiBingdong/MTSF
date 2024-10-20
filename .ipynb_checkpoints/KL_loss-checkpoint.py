import sys
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from numpy import random
from skimage.transform import resize


def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss1vs1(nn.Module):
    def __init__(self, dev='cuda'):
        super(KLDLoss1vs1, self).__init__()
        self.dev=dev

    def KLD(self, inp, trg):
        assert inp.size(0)==trg.size(0), "Sizes of the distributions doesn't match"
        batch_size=inp.size(0)
        kld_tensor=torch.empty(batch_size)
        for k in range(batch_size):
            eps = sys.float_info.epsilon
            i = inp[k]/(torch.sum(inp[k])+eps)
            t = trg[k]/(torch.sum(trg[k])+eps)
            #eps = sys.float_info.epsilon
            kld_tensor[k]= torch.sum(t*torch.log(eps+torch.div(t,(i+eps))))
        return kld_tensor.to(self.dev)

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)

def CC(saliency_map_1, saliency_map_2):
    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 'Nan' # or change to mean value
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]
def cc_numeric(y_true, y_pred):
    """
    Function to evaluate Pearson's correlation coefficient (sec 4.2.2 of [1]) on two samples.
    The two distributions are numpy arrays having arbitrary but coherent shapes.

    :param y_true: groundtruth.
    :param y_pred: predictions.
    :return: numeric cc.
    """
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    batch_size = y_true.shape[0]
    #print(batch_size)
    b=0.0
    for i in range(batch_size):
        a = CC(y_true, y_pred)
        if str(a) =='nan':
            print('no data')
            a='Nan' # or change to mean value
        b = b + a
    return b/batch_size


def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value

    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        print('no data')
        saliency_map[:] = 'Nan' # or change to mean value
        saliency_map /= saliency_map.sum()

    return saliency_map

def SIM_numeric(saliency_map_1, saliency_map_2):
    """ Compute similiarity metric. """

    saliency_map_1 = saliency_map_1.cpu().detach().numpy()
    saliency_map_2 = saliency_map_2.cpu().detach().numpy()
    batch_size = saliency_map_1.shape[0]
    #print(batch_size)
    b=0.0
    for i in range(batch_size):
        density_1 = convert_saliency_map_to_density(saliency_map_1[i], minimum_value=0)
        density_2 = convert_saliency_map_to_density(saliency_map_2[i], minimum_value=0)

        a= np.min([density_1, density_2], axis=0).sum()
        if str(a) =='nan':
            print('no data')
            a = 'Nan' # or change to mean value
        b = b + a
    return b/batch_size


def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)

    # If there are no fixations to predict, return NaN
    saliencyMap = saliencyMap.cpu().detach().numpy()
    fixationMap = fixationMap.cpu().detach().numpy()
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        if str(score) == 'nan':
            print('no data')
            score = 'Nan' # or change to mean value
        return score

    # make sure maps have the same shape
    #from scipy.misc import imresize
    #map1 = imresize(saliencyMap, np.shape(fixationMap))
    batch_size = saliencyMap.shape[0]
    b=0.0
    for i in range(batch_size):
        map1 = saliencyMap[i,:,:,:]
        if not map1.max() == 0:
            map1 = map1.astype(float) / map1.max()

        # normalize saliency map
        if not map1.std(ddof=1) == 0:
            map1 = (map1 - map1.mean()) / map1.std(ddof=1)

        # mean value at fixation locations
        map2 = fixationMap[i,:,...]
        #score = map1[fixationMap.astype(bool)].mean()
        score = map1[map2.astype(bool)].mean()
        if str(score) =='nan':
            score = 'Nan' # or change to mean value
        b= b+ score
    return b/batch_size

def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve
    saliencyMap = saliencyMap.cpu().detach().numpy()
    fixationMap = fixationMap.cpu().detach().numpy()
    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        if str(score) == 'nan':
            score = 'Nan' # or change to mean value
        return score


    '''
    # make the saliencyMap the size of the image of fixationMap
    
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))
    '''

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan') # or change to mean value
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score
def AUC_Judd_batch(saliencyMap, fixationMap):
    batch_size = saliencyMap.shape[0]
    b = 0.0
    for i in range(batch_size):
        salmap = saliencyMap[i,...]
        fixmap = fixationMap[i,...]
        score = AUC_Judd(salmap,fixmap)
        b = b+ score
    return b/batch_size

def AUC_shuffled(saliencyMap, fixationMap, otherMap, Nsplits=100, stepSize=0.1, toPlot=False):
    '''saliencyMap is the saliency map
    fixationMap is the human fixation map (binary matrix)
    otherMap is a binary fixation map (like fixationMap) by taking the union of
    fixations from M other random images (Borji uses M=10)
    Nsplits is number of random splits
    stepSize is for sweeping through saliency map
    if toPlot=1, displays ROC curve
    '''

    # saliencyMap = saliencyMap.transpose()
    # fixationMap = fixationMap.transpose()
    # otherMap = otherMap.transpose()
    saliencyMap = saliencyMap.cpu().detach().numpy()
    fixationMap = fixationMap.cpu().detach().numpy()

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        if str(score) == 'nan':
            score = 'Nan' # or change to mean value
        return score

    '''
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        saliencyMap = np.array(Image.fromarray(saliencyMap).resize((np.shape(fixationMap)[1], np.shape(fixationMap)[0])))
    '''

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    S = saliencyMap.flatten(order='F')
    F = fixationMap.flatten(order='F')
    Oth = otherMap.flatten(order='F')

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations specified by otherMap
    ind = np.nonzero(Oth)[0] # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.empty((Nfixations_oth,Nsplits))
    randfix[:] = np.nan

    for i in range(Nsplits):
        randind = ind[np.random.permutation(len(ind))]  # randomize choice of fixation locations
        randfix[:, i] = S[randind[:Nfixations_oth]] # sal map values at random fixation locations of other random images

    # calculate AUC per random split (set of random locations)
    auc = np.empty(Nsplits)
    auc[:] = np.nan

    def Matlab_like_gen(start, stop, step, precision):
        r = start
        while round(r, precision) <= stop:
            yield round(r, precision)
            r += step

    for s in range(Nsplits):
        curfix = randfix[:, s]
        i0 = Matlab_like_gen(0, max(np.maximum(Sth, curfix)), stepSize, 5)
        allthreshes = [x for x in i0]
        allthreshes.reverse()

        tp = np.zeros((len(allthreshes) + 2))
        fp = np.zeros((len(allthreshes) + 2))
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i+1] = (Sth >= thresh).sum() / Nfixations
            fp[i+1] = (curfix >= thresh).sum() / Nfixations_oth

        auc[s] = np.trapz(tp, x=fp)

    score = np.mean(auc)  # mean across random splits

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score


'''created: Zoya Bylinskii, March 6
based on: Kummerer et al.
(http://www.pnas.org/content/112/52/16054.abstract)
Python Implementation: Lapo Faggi, Oct 2020

This finds the information-gain of the saliencyMap over a baselineMap'''


def InfoGain(saliencyMap, fixationMap, baselineMap):
    '''saliencyMap is the saliency map
    fixationMap is the human fixation map (binary matrix)
    baselineMap is another saliency map (e.g. all fixations from other images)'''

    saliencyMap = saliencyMap.cpu().detach().numpy()
    fixationMap = fixationMap.cpu().detach().numpy()
    #baselineMap = baselineMap.cpu().detach().numpy()
    batch_size = saliencyMap.shape[0]
    b = 0.0
    for i in range(batch_size):
        map1 = saliencyMap[i,...]
        mapb = baselineMap

        #map1 = np.resize(saliencyMap,np.shape(fixationMap))
        #mapb = np.resize(baselineMap, np.shape(fixationMap))

        # normalize and vectorize saliency maps
        map1 = (map1.flatten(order='F') - np.min(map1))/ (np.max(map1 - np.min(map1)))
        mapb = (mapb.flatten(order='F') - np.min(mapb))/(np.max(mapb - np.min(mapb)))

        # turn into distributions
        map1 /= np.sum(map1)
        mapb /= np.sum(mapb)
        mapfix = fixationMap[i,...]
        mapfix = mapfix.flatten(order = 'F')
        locs = mapfix > 0

        eps = 2.2204e-16
        score = np.mean(np.log2(eps+map1[locs])-np.log2(eps+mapb[locs]))
        if str(score)=='nan':
            score = 'Nan' # or change to mean value
        b = b+ score
    return b/batch_size

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
         Human fixation map.
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    rand_sampler : callable
        S_rand = rand_sampler(S, F, n_rep, n_fix)
        Sample the saliency map at random locations to estimate false positive.
        Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
    Returns
    -------
    AUC : float, between [0,1]
    '''

    #saliency_map = np.array(saliency_map, copy=False)
    #fixation_map = np.array(fixation_map, copy=False) > 0.5
    saliency_map = saliency_map.cpu().detach().numpy()
    fixation_map = fixation_map.cpu().detach().numpy()
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        score = float('nan')
        if str(score) == 'nan':
            score = 'Nan' # or change to mean value
        return score
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    #saliency_map = normalize(saliency_map, method='range')
    saliency_map = (saliency_map - saliency_map.min()) \
                  / (saliency_map.max() - saliency_map.min())

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    score = np.mean(auc)
    return score # Average across random split

def AUC_Borji_batch(saliency_map, fixation_map):
    batch_size = saliency_map.shape[0]
    b = 0.0
    for i in range(batch_size):
        salmap = saliency_map[i, ...]
        fixmap = fixation_map[i, ...]
        score = AUC_Borji(salmap, fixmap)
        b = b + score
    return b / batch_size
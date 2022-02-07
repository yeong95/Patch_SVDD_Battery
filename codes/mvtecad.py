import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

DATASET_PATH = '/workspace/CAMPUS/TOFU_Box/' # test용
DATASET_PATH_origin = '/workspace/CAMPUS/TOFU_Box/' # Train용


CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']



__all__ = ['objs', 'set_root_path',
           'get_x', 'get_x_standardized',
           'detection_auroc', 'segmentation_auroc']


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def bilinears(images, shape) -> np.ndarray:
    import cv2
    N = images.shape[0]
    new_shape = (N,) + shape
    ret = np.zeros(new_shape, dtype=images.dtype)
    for i in range(N):
        ret[i] = cv2.resize(images[i], dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
    return ret


def gray2rgb(images):
    tile_shape = tuple(np.ones(len(images.shape), dtype=int))
    tile_shape += (3,)

    images = np.tile(np.expand_dims(images, axis=-1), tile_shape)
    # print(images.shape)
    return images


def set_root_path(new_path):
    global DATASET_PATH
    DATASET_PATH = new_path


def get_x(mode='train'):

    if mode == 'test':
        
        fpattern1 = os.path.join(DATASET_PATH,  f'{mode}/NG/*/*.jpg') # NG가 먼저 들어가고
        fpaths1 = sorted(glob(fpattern1))
        print("Anomaly test image: ", len(fpaths1), " 장")
        print("경로 확인: ", fpaths1[0])
        
        fpattern2 = os.path.join(DATASET_PATH,  f'{mode}/OK/*/*.jpg') # OK는 그 다음에 들어감
        fpaths2 = sorted(glob(fpattern2))
        print("Normal test image: ", len(fpaths2), " 장")
        print("경로 확인: ", fpaths2[0])
        
        images1 = np.asarray(list(map(imread, fpaths1)))
        images2 = np.asarray(list(map(imread, fpaths2)))
        images = np.concatenate([images1, images2])

    else:
        
        fpattern1 = os.path.join(DATASET_PATH_origin, f'{mode}/OK/정상A/*.jpg') # 2236장
        #fpaths1 = sorted(glob(fpattern1))[:745] # 1/3
        #fpaths1 = sorted(glob(fpattern1))[:1118] # 1/2
        #fpaths1 = sorted(glob(fpattern1))[:1490] # 2/3
        fpaths1 = sorted(glob(fpattern1)) # 전체
        print("정상 A: ", len(fpaths1), " 장")
        print("경로 확인: ", fpaths1[0])
    
        fpattern2 = os.path.join(DATASET_PATH_origin, f'{mode}/OK/정상B/*.jpg') # 1430장
        #fpaths2 = sorted(glob(fpattern2))[:476] # 1/3
        #fpaths2 = sorted(glob(fpattern2))[:715] # 1/2
        #fpaths2 = sorted(glob(fpattern2))[:953] # 2/3
        fpaths2 = sorted(glob(fpattern2)) # 전체
        print("정상 B: ", len(fpaths2), " 장")
        print("경로 확인: ", fpaths2[0])
        
        images1 = np.asarray(list(map(imread, fpaths1)))
        images2 = np.asarray(list(map(imread, fpaths2)))
        images = np.concatenate([images1, images2])

    if images.shape[-1] != 3:
        images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images)
    return images


def get_x_standardized(mode='train'):
    x = get_x(mode=mode)
    mean = get_mean()
    return (x.astype(np.float32) - mean) / 255


def get_label():
    mode = 'test'
    
    fpattern1 = os.path.join(DATASET_PATH,  f'{mode}/NG/*/*.jpg')
    fpaths1 = sorted(glob(fpattern1))

    fpattern2 = os.path.join(DATASET_PATH,  f'{mode}/OK/*/*.jpg')
    fpaths2 = sorted(glob(fpattern2))
    
    
    Nanomaly = len(fpaths1)
    Nnormal = len(fpaths2)
    labels = np.zeros(Nanomaly + Nnormal, dtype=np.int32)
    labels[:Nanomaly] = 1
    return labels



def get_mean():
    images = get_x(mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean


def detection_auroc(anomaly_scores):
    label = get_label()  # 1: anomaly 0: normal
    auroc = roc_auc_score(label, anomaly_scores)
    return auroc



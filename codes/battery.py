import numpy as np
from PIL import Image
from imageio import imread
from glob import glob
from sklearn.metrics import roc_auc_score
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T




# CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                'tile', 'toothbrush', 'transistor', 'wood', 'zipper']



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


def get_x(mode='train', args=None):

    if mode == 'test':
        
        fpattern = os.path.join(args.dataset_path, f'{mode}/*/*.bmp')
        fpaths = sorted(glob(fpattern))
        print("test image: ", len(fpaths), " 장")
#         print("경로 확인: ", fpaths[0])
        
        images = np.asarray(list(map(imread, fpaths)))
        
    else:        
        fpattern = os.path.join(args.dataset_path, f'{mode}/*/*.bmp')
#         print(fpattern)
        fpaths = sorted(glob(fpattern))
        print("train image: ", len(fpaths), " 장")
#         print("경로 확인: ", fpaths[0])
        
        images = np.asarray(list(map(imread, fpaths)))
        

    if images.shape[-1] != 3:
        images = gray2rgb(images)
    images = list(map(resize, images))
    images = np.asarray(images)
    return images


def get_x_standardized(mode='train', args=None):
    x = get_x(mode=mode, args=args)
    mean = get_mean(x)
    return (x.astype(np.float32) - mean) / 255


def get_label(args):
    mode = 'test'
    
    if args.dataset_type == 'multi':
        normal_names = ['코팅부 경계부 불량', '무지부 줄무늬', '코팅부 접힘', '코팅부 미코팅', '코팅부 줄무늬', '코팅부 테이프', \
                     '코팅부 기재연결부', '무지부 기재연결부', '코팅부 코팅불량']
        abnormal_names = ['코팅부 버블', '코팅부 흑점', '무지부 주름', '코팅부 찍힘', '코팅부 백점', '코팅부 라벨지']
    elif args.dataset_type == 'one':
        normal_names = ['코팅부 경계부 불량']
        abnormal_names = ['무지부 줄무늬', '코팅부 접힘', '코팅부 미코팅', '코팅부 줄무늬', '코팅부 테이프', \
                     '코팅부 기재연결부', '무지부 기재연결부', '코팅부 코팅불량', '코팅부 버블', '코팅부 흑점', '무지부 주름', '코팅부 찍힘', '코팅부 백점', '코팅부 라벨지']
    
#     normal_paths = []
#     for name in normal_names:
#         fpattern = os.path.join(DATASET_PATH, f'{mode}/{name}/*.bmp')
#         fpaths = sorted(glob(fpattern))
# #         print(len(fpaths))
#         normal_paths += fpaths
    
#     abnormal_paths = []
#     for name in abnormal_names:
#         fpattern = os.path.join(DATASET_PATH, f'{mode}/{name}/*.bmp')
#         fpaths = sorted(glob(fpattern))
# #         print(len(fpaths))
#         abnormal_paths += fpaths

    fpattern = os.path.join(args.dataset_path, f'{mode}/*/*.bmp')
    fpaths = sorted(glob(fpattern))
    
    # get only label name in data path 
    def split(x):
        return x.split('/')[-2]    

    test_class_list = list(map(split, fpaths))
    
    def change_name_to_label(x):
        if x in normal_names:
            return 0
        else:
            return 1

    test_label_list = list(map(change_name_to_label, test_class_list))

    return test_label_list, test_class_list



def get_mean(images):
#     images = get_x(mode='train')
    mean = images.astype(np.float32).mean(axis=0)
    return mean


def detection_auroc(anomaly_scores, args):
    label, classes = get_label(args)  # 1: anomaly 0: normal
    auroc = roc_auc_score(label, anomaly_scores)
    return auroc



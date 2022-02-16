from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores, distribute_score
from torchvision.models import resnet18, resnet34
from tqdm import tqdm
from random import sample
import os, sys, humanize, psutil, GPUtil
import gc
import torch.nn.functional as F
import torch.nn as nn
from numba import jit
from efficientnet_pytorch import EfficientNet
import codes.mvtecad as mvtec
import codes.battery as battery
import time
import scipy.stats

import optuna
from optuna.trial import TrialState


@jit
def train_distribution(cov, emb, num_range, I):
    for i in range(num_range):
        cov[i, :, :] = np.cov(emb[:, i, :], rowvar = False) + 0.01 * I    # np.cov(emb[:, i, :]): batch, 344)에 대한 covariance 구함 -> (344,344)
    return cov

@jit
def mahal(u, v, VI):
    VI = np.atleast_2d(VI)
    delta = u - v
    p1 = np.dot(delta, VI)
    m = np.dot(p1, delta)
    return np.sqrt(m)


@jit
def test_inference(mean, cov, num_range, emb):

    dist_list = []

    for i in range(num_range):
        partial_mean = mean[i, :]
        conv_inv = np.linalg.inv(cov[i, :, :])
        dist = [mahal(sample[i, :], partial_mean, conv_inv) for sample in emb]
        dist_list.append(dist)

    return dist_list



# Define function
def mem_report(word):
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
    #GPUs = GPUtil.getGPUs()
    #for i, gpu in enumerate(GPUs):
    #    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x_tr, x_te, K, S, models):
    
    #============ResNet Backbone===============#

#     outputs = []

#     def hook(module, input, output):
#        outputs.append(F.adaptive_avg_pool2d(output.detach().cpu(), output_size = (1, 1)))

#     models.layer1[-1].register_forward_hook(hook)
#     models.layer2[-1].register_forward_hook(hook)
#     models.layer3[-1].register_forward_hook(hook)

    
    x_tr = NHWC2NCHW(x_tr) # change to (batch, C, H, W)
    dataset_tr = PatchDataset_NCHW(x_tr, K=K, S=S)
    loader_tr = DataLoader(dataset_tr, batch_size=512, shuffle=False, pin_memory=False)
    embs_tr = np.empty((dataset_tr.N, dataset_tr.row_num, dataset_tr.col_num, 344), dtype=np.float32)  # [-1, I, J, D]
    # Effb0- 192dim, Effb1- 192dim, Effb2- 208dim, Effb3 - 240dim, Effb4 - 272dim, Effb5 - 304dim, Effb6 - 344dim, Effb7 - 384dim
    # ResNet18 - 448dim, ResNet34 - 448dim
    
    with torch.no_grad():
        for idx, (xs, ns, iis, js) in enumerate(tqdm(loader_tr, "Train infer %dx%d patch" %(K, K) , position = 0, leave = True)):
            xs = xs.cuda() # (512, 3, 64, 64)
            
            #_ = models(xs)

            #embedding = torch.cat((outputs[0], outputs[1], outputs[2]), dim = 1) # (64+128+256 => 512, 448, 1, 1 )
            
            # Prediction
            endpoints = models.extract_endpoints(xs)

            del xs



            embedding = torch.cat((F.adaptive_avg_pool2d(endpoints['reduction_1'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_2'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_3'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_4'].detach().cpu(), output_size = (1, 1))), dim = 1)


            outputs = []

            for embed, n, i, j in zip(embedding, ns, iis, js):  # embedding : [512, 344, 1, 1], n:image number i,j: patch location  
                embs_tr[n, i, j] = np.squeeze(embed)  #embed: [344, 1, 1]

            del endpoints
            del embedding

    del x_tr # 삭제
    

    x_te = NHWC2NCHW(x_te)
    dataset_te = PatchDataset_NCHW(x_te, K=K, S=S)
    loader_te = DataLoader(dataset_te, batch_size=512, shuffle=False, pin_memory=False)
    embs_te = np.empty((dataset_te.N, dataset_te.row_num, dataset_te.col_num, 344), dtype=np.float32)  # [-1, I, J, D]
    with torch.no_grad():
        for idx, (xs, ns, iis, js) in enumerate(tqdm(loader_te, "test infer %dx%d patch" % (K, K), position = 0, leave = True)):
            xs = xs.cuda() # (64, 3, 64, 64)

            # Prediction
            endpoints = models.extract_endpoints(xs)
            
            
            
            #_ = models(xs)

            #del xs
            
            #embedding = torch.cat((outputs[0], outputs[1], outputs[2]), dim = 1) # (64+128+256 => 512, 448, 1, 1 )


             
            embedding = torch.cat((F.adaptive_avg_pool2d(endpoints['reduction_1'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_2'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_3'].detach().cpu(), output_size = (1, 1)),
                                   F.adaptive_avg_pool2d(endpoints['reduction_4'].detach().cpu(), output_size = (1, 1))), dim = 1)

            
            outputs = []

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs_te[n, i, j] = np.squeeze(embed)

            del embedding
            del endpoints
    
    del x_te
    
    mem_report("Before Train distribution")

    B_train, H_train, W_train, C_train = embs_tr.shape   # embs_tr: [batch, 9, 9, 344]
    emb_train = embs_tr.reshape((B_train, H_train * W_train, C_train))  

    del embs_tr

    mean_train = np.mean(emb_train, axis = 0)   # mean_train: (81, 344)
    cov_train = np.zeros((H_train * W_train, C_train, C_train))   # cov_train: (81, 344, 344)

    I = np.identity(C_train) # I: (344, 344)

    # covariance 계산을 numba로 처리.
    cov_train = train_distribution(cov_train, emb_train, H_train*W_train, I)
    

    del emb_train  
    del I

    mem_report("After Train distribution") # memory usage checking.

    B_test, H_test, W_test, C_test = embs_te.shape 
    emb_test = embs_te.reshape((B_test, H_test * W_test, C_test))   # (batch, 9*9, 344)

    del embs_te

    #dist_list = []

    cov_train = cov_train.astype('float32')

    
#     train_dist_list = test_inference
    # Test inference를 대체
    dist_list = test_inference(mean_train, cov_train, H_test*W_test, emb_test)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B_test, H_test, W_test)   # (batch, 9, 9)
    
    mem_report("After Test Inference") # memory usage checking.
    
    
    if K == 64:
        mean_64_sum = np.sum(mean_train, axis = -1)   # [81, 344] -> [81]
        mean_64_size = int(mean_train.shape[0] ** 0.5)  # 9 
        mean_64_KL = mean_64_sum.reshape(mean_64_size, mean_64_size)
        mean_64_min = np.min(mean_64_KL)
        mean_64_KL = mean_64_KL - mean_64_min
        normality_map = distribute_score(mean_64_KL, (256, 256), K=64, S=24)
    else:
        mean_32_sum = np.sum(mean_train, axis = -1)
        mean_32_size = int(mean_train.shape[0] ** 0.5)
        mean_32_KL = mean_32_sum.reshape(mean_32_size, mean_32_size)
        mean_32_min = np.min(mean_32_KL)
        mean_32_KL = mean_32_KL - mean_32_min
        normality_map = distribute_score(mean_32_KL, (256, 256), K=32, S=14)
    

    return dist_list, normality_map





def assess_anomaly_maps(trial, anomaly_maps, category, normal_map, args):
    
    lambda_opt = trial.suggest_int('lambda', 0, 100000)
    
    print("Anomaly Map shape: ", anomaly_maps.shape)
    
    anomaly_kl_scores = np.zeros(anomaly_maps.shape[0])
    
    for i in range(anomaly_maps.shape[0]):
        anomaly_kl_scores[i] = scipy.stats.entropy(normal_map.flatten(), anomaly_maps[i].flatten())
        
#     Det_AUROC = dict()
    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    
    total_anomaly_scores = anomaly_scores + lambda_opt * anomaly_kl_scores
    
    auroc_det = battery.detection_auroc(total_anomaly_socres, args)

    return auroc_det


#########################

def eval_encoder_NN_multiK(args):

#     x_tr = mvtecad.get_x_standardized(mode='train') # Training dataset
#     x_te = mvtecad.get_x_standardized(mode='test')  # Test dataset

    x_tr = battery.get_x_standardized(mode='train', args=args)
    x_te = battery.get_x_standardized(mode='test', args=args)

    print("dataset completed !")

    #========ResNet backbone========#
    #model = resnet34(pretrained=True) # Resnet 사용 시
    
    if args.load_pretrained_model:
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=15) # EfficientNet 사용 시 
#         model.load_state_dict(torch.load(args.pretrained_model_path))
#         import pdb;pdb.set_trace()
    
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)

        model.to(device)
        print("load pretrained efficientnet model...")        
    else:
        model = EfficientNet.from_pretrained('efficientnet-b6') # EfficientNet 사용 시 
        model.to(device)
        model.eval()

    load_map = args.load_map
    if not load_map:
        maps_64, normality_map_64 = infer(x_tr, x_te, K=64, S=24, models=model)
        maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=24)
        #det_64 = assess_anomaly_maps(maps_64, "det_64")

        maps_32, normality_map_32 = infer(x_tr, x_te, K=32, S=14, models=model)
        maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=14) 
        #det_32 = assess_anomaly_maps(maps_32, "det_32")
        
        # save 
        np.save(f'{args.save_path}/maps_64.npy', maps_64)
        np.save(f'{args.save_path}/normality_map_64.npy', normality_map_64)
        
        np.save(f'{args.save_path}/maps_32.npy', maps_32)
        np.save(f'{args.save_path}/normality_map_32.npy', normality_map_32)
        
    else:
        maps_64 = np.load("maps_64.npy")
        normality_map_64 = np.load("normality_map_64.npy")
        
        maps_32 = np.load("maps_32.npy")
        normality_map_32 = np.load("normality_map_32.npy")
        
        print("load trained map...")

    
    
    maps_mult = maps_64 * maps_32
    Final_normality_map = normality_map_64 * normality_map_32
    det_mult = assess_anomaly_maps(maps_mult, "det_mult", Final_normality_map, args)

    return {
        #'det_64': det_64,
        #'det_32': det_32,
        'det_mult': det_mult,

        #'maps_64': maps_64,
        #'maps_32': maps_32,
        'maps_mult': maps_mult,
    }


#########################################################################################


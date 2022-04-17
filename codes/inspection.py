from codes import mvtecad
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores, distribute_score, save_pickle
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
import random

# for reproducible experiment 
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

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
device = torch.device('cuda:1' if use_cuda else 'cpu')


__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']


def infer(x_tr, x_te, K, S, models, args):
    
    #============ResNet Backbone===============#

#     outputs = []

#     def hook(module, input, output):
#        outputs.append(F.adaptive_avg_pool2d(output.detach().cpu(), output_size = (1, 1)))

#     models.layer1[-1].register_forward_hook(hook)
#     models.layer2[-1].register_forward_hook(hook)
#     models.layer3[-1].register_forward_hook(hook)

    
    x_tr = NHWC2NCHW(x_tr) # change to (batch, C, H, W)
    dataset_tr = PatchDataset_NCHW(x_tr, K=K, S=S)
    loader_tr = DataLoader(dataset_tr, batch_size=256, shuffle=False, pin_memory=False)
    embs_tr = np.empty((dataset_tr.N, dataset_tr.row_num, dataset_tr.col_num, 344), dtype=np.float32)  # [-1, I, J, D]
    # Effb0- 192dim, Effb1- 192dim, Effb2- 208dim, Effb3 - 240dim, Effb4 - 272dim, Effb5 - 304dim, Effb6 - 344dim, Effb7 - 384dim
    # ResNet18 - 448dim, ResNet34 - 448dim
    try:
        embs_tr = np.load(f"{args.load_path}/{K}_embs_tr_{args.idx_}.npy")
        print("load embs_tr")
    except:
        with torch.no_grad():
            for idx, (xs, ns, iis, js) in enumerate(tqdm(loader_tr, "Train infer %dx%d patch" %(K, K) , position = 0, leave = True)):
                xs = xs.to(device) # (512, 3, 64, 64)

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
        
        np.save(f"{args.save_path}/{K}_embs_tr_{args.idx_}", embs_tr)

    
    x_te = NHWC2NCHW(x_te)
    dataset_te = PatchDataset_NCHW(x_te, K=K, S=S)
    loader_te = DataLoader(dataset_te, batch_size=256, shuffle=False, pin_memory=False)
    embs_te = np.empty((dataset_te.N, dataset_te.row_num, dataset_te.col_num, 344), dtype=np.float32)  # [-1, I, J, D]
    
    try:
        embs_te = np.load(f"{args.load_path}/{args.mode}_{K}_embs_te_{args.idx_}.npy")
        print("load embs_te")
    except:
        with torch.no_grad():
            for idx, (xs, ns, iis, js) in enumerate(tqdm(loader_te, "test infer %dx%d patch" % (K, K), position = 0, leave = True)):
                xs = xs.to(device) # (64, 3, 64, 64)

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
        
        np.save(f"{args.save_path}/{args.mode}_{K}_embs_te_{args.idx_}", embs_te)
    
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

#     mem_report("After Train distribution") # memory usage checking.

    B_test, H_test, W_test, C_test = embs_te.shape 
    emb_test = embs_te.reshape((B_test, H_test * W_test, C_test))   # (batch, 9*9, 344)

    del embs_te

    #dist_list = []

    cov_train = cov_train.astype('float32')

    
#     train_dist_list = test_inference
    # Test inference를 대체
    dist_list = test_inference(mean_train, cov_train, H_test*W_test, emb_test)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B_test, H_test, W_test)   # (batch, 9, 9)
    
#     mem_report("After Test Inference") # memory usage checking.

    return dist_list



def assess_anomaly_maps(anomaly_maps, category, args):
    print("Anomaly Map shape: ", anomaly_maps.shape)
    
    Det_AUROC = dict()
    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    np.save(f"{args.save_path}/{args.mode}_{category}_anomaly_scores", anomaly_scores)
    final_det_auroc = 0 

    return final_det_auroc


#########################

def inference_procedure(x_tr, x_te, model, idx_=1, args=None):
    
    args.idx_ = idx_
    if idx_ <  100:
        maps_256 = infer(x_tr, x_te, K=256, S=64, models=model, args=args)
        maps_256 = distribute_scores(maps_256, (256, 256), K=256, S=64)

        maps_32 = infer(x_tr, x_te, K=32, S=14, models=model, args=args)
        maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=14) 

    else:
        maps_64 = np.load(f"save/무지부_코팅부/3_13_efficientnet_b6/test_maps_64_{idx_}.npy")
        normality_map_64 = np.load(f"save/무지부_코팅부/3_13_efficientnet_b6/normality_map_64_{idx_}.npy")
        
        maps_32 = np.load(f"save/무지부_코팅부/3_13_efficientnet_b6/test_maps_32_{idx_}.npy")
        normality_map_32 = np.load(f"save/무지부_코팅부/3_13_efficientnet_b6/normality_map_32_{idx_}.npy")
        
        print("load trained map...")

    # save 
    if not os.path.isfile(f'{args.save_path}/{args.mode}_maps_256_{idx_}.npy'):
        np.save(f'{args.save_path}/{args.mode}_maps_256_{idx_}.npy', maps_256)
    
    if not os.path.isfile(f'{args.save_path}/{args.mode}_maps_32_{idx_}.npy'):
        np.save(f'{args.save_path}/{args.mode}_maps_32_{idx_}.npy', maps_32)
    
    maps_mult = maps_256 * maps_32
    
    args.dataset_type = f'model_{idx_}'
    det_mult = assess_anomaly_maps(maps_mult, f"model_{idx_}", args)
    
    return det_mult
    

    
def compute_final_anomaly_score(args):
    
#     for i in range(1,5):
#         globals()['score_{}' .format(i)] = np.load(f'save/four ensemble model/2_8_efficientnet_b6/model_{i}_anomaly_scores.npy')
    
    for i in range(1,6):
        globals()['score_{}' .format(i)] = np.load(f'{args.save_path}/model_{i}_anomaly_scores.npy')
    
#     import pdb;pdb.set_trace()
    new_score = []
    for i in range(len(score_1)):
        new_score.append(min(score_1[i], score_2[i], score_3[i], score_4[i], score_5[i], score_6[i]))
#     pdb.set_trace()
    
    save_path = f'{args.save_path}/final_ano_score.pickle'
    save_pickle(save_path, new_score)
        
        
    
    
def eval_encoder_NN_multiK(args):

    x_tr_1 = battery.get_x_standardized(mode='train', args=args, normal_class = args.normal_class[0])  # model 1
    x_tr_2 = battery.get_x_standardized(mode='train', args=args, normal_class = args.normal_class[1])  # model 2
    x_tr_3 = battery.get_x_standardized(mode='train', args=args, normal_class = args.normal_class[2])  # model 3
    x_tr_4 = battery.get_x_standardized(mode='train', args=args, normal_class = args.normal_class[3])  # model 4
    x_tr_5 = battery.get_x_standardized(mode='train', args=args, normal_class = args.normal_class[4])  # model 5

    # load validation : train + valid 
    x_valid_1 = battery.get_x_standardized(mode='valid', args=args, normal_class = args.normal_class[0])  # model 1
    x_valid_2 = battery.get_x_standardized(mode='valid', args=args, normal_class = args.normal_class[1])  # model 2
    x_valid_3 = battery.get_x_standardized(mode='valid', args=args, normal_class = args.normal_class[2])  # model 3
    x_valid_4 = battery.get_x_standardized(mode='valid', args=args, normal_class = args.normal_class[3])  # model 4
    x_valid_5 = battery.get_x_standardized(mode='valid', args=args, normal_class = args.normal_class[4])  # model 5
    
    x_te = battery.get_x_standardized(mode='test', args=args)

    print("dataset completed !")

    #========ResNet backbone========#
    #model = resnet34(pretrained=True) # Resnet 사용 시
    
    if args.load_pretrained_model:
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=15) # EfficientNet 사용 시 
    
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
    
    # validation data 이용하여 정상 분포 구하기 
    args.mode = 'valid'
    print("validation start")
    print("model 1 start...")
    det_mult = inference_procedure(x_tr_1, x_valid_1, model, idx_=1, args=args)
    print("model 2 start...")
    det_mult = inference_procedure(x_tr_2, x_valid_2, model, idx_=2, args=args)
    print("model 3 start...")
    det_mult = inference_procedure(x_tr_3, x_valid_3, model, idx_=3, args=args)
    print("model 4 start...")
    det_mult = inference_procedure(x_tr_4, x_valid_4, model, idx_=4, args=args)
    print("model 5 start...")
    det_mult = inference_procedure(x_tr_5, x_valid_5, model, idx_=5, args=args)
       
    args.mode = 'test'
    print("test start")
    print("model 1 start...")
    det_mult = inference_procedure(x_tr_1, x_te, model, idx_=1, args=args)
    print("model 2 start...")
    det_mult = inference_procedure(x_tr_2, x_te, model, idx_=2, args=args)
    print("model 3 start...")
    det_mult = inference_procedure(x_tr_3, x_te, model, idx_=3, args=args)
    print("model 4 start...")
    det_mult = inference_procedure(x_tr_4, x_te, model, idx_=4, args=args)
    print("model 5 start...")
    det_mult = inference_procedure(x_tr_5, x_te, model, idx_=5, args=args)


    # put test image and compute anomaly score
#     compute_final_anomaly_score(args)
#     det_mult = 0
    

    return {
        #'det_64': det_64,
        #'det_32': det_32,
        'det_mult': det_mult,

#         'maps_64': maps_64,
#         'maps_32': maps_32,
#         'maps_mult': maps_mult,
    }


#########################################################################################


from codes.utils import save_pickle
import argparse

from datetime import datetime
import os

from codes.inspection import eval_encoder_NN_multiK


def do_evaluate_encoder_multiK(args):
    results = eval_encoder_NN_multiK(args)

    #det_64 = results['det_64']
    #det_32 = results['det_32']

    #det_sum = results['det_sum']
#     det_mult = results['det_mult']
    
#     save_pickle(f'{args.save_path}/results.pickle', results)

#     print(
#         f'| mult | Det: {det_mult:.3f}')


#########################


def main():
    parser = argparse.ArgumentParser(description="Patch SVDD")
    
#     parser.add_argument("--dataset_type", type=str, default='one', choices=['model_1', 'model_2', 'model_3', 'model_4'])
    parser.add_argument("--load_map", type=bool, default=False)
    parser.add_argument("--load_pretrained_model", type=bool, default=False)
    parser.add_argument("--pretrained_model_name", type=str)
    
    args = parser.parse_args()    
    
    month = datetime.today().month
    day = datetime.today().day
#     hour = datetime.now().hour

    # save path
    args.save_path = f'./save/ensemble model/{month}_{day}_efficientnet_b6'
    os.makedirs(args.save_path, exist_ok=True)
    
    # pretrained model path 
    args.pretrained_model_path = f'./save/finetune/{args.pretrained_model_name}'
    
    # data path
    args.dataset_path = f'/tf/KAIER_2022/Battery_data/eight ensemble model'        

#     DATASET_PATH = '/tf/KAIER_2022/Battery_data/multi_class' # test용
#     DATASET_PATH_origin = '/tf/KAIER_2022/Battery_data/multi_class' # Train용
    
    # normal class
    args.normal_class = ['코팅부 경계부 불량','무지부 줄무늬', '코팅부 접힘', '코팅부 미코팅', '코팅부 줄무늬', '코팅부 테이프', '코팅부 기재연결부', '무지부 기재연결부']
    
    do_evaluate_encoder_multiK(args)


if __name__ == '__main__':
    main()

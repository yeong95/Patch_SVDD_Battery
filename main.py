from codes.utils import save_pickle
import argparse

from datetime import datetime
import os

from codes.inspection import eval_encoder_NN_multiK
from pytz import timezone

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
#     parser.add_argument("--data_choice", type=str, default='무지부', choices=['무지부', '코팅부'])
    
    args = parser.parse_args()    
    
    korea_date = datetime.now(timezone('Asia/Seoul'))

    month = korea_date.month
    day = korea_date.day
#     hour = datetime.now().hour

    # save path
    args.save_path = f'./save/무지부_코팅부/{month}_{day}_efficientnet_b6'
    args.load_path = args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    
    # pretrained model path 
    args.pretrained_model_path = f'./save/finetune/{args.pretrained_model_name}'
    
    # data path
    args.dataset_path = f'/tf/Battery_data/무지부_코팅부_테이프제외2/코팅부'        

#     DATASET_PATH = '/tf/KAIER_2022/Battery_data/multi_class' # test용
#     DATASET_PATH_origin = '/tf/KAIER_2022/Battery_data/multi_class' # Train용
    
    # normal class
    args.normal_class = ['코팅부 경계부 불량', '코팅부 접힘', '코팅부 미코팅', '코팅부 줄무늬', '코팅부 코팅불량']
    
    do_evaluate_encoder_multiK(args)


if __name__ == '__main__':
    main()

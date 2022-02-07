import matplotlib.pyplot as plt
from codes import mvtecad
from codes import battery
from tqdm import tqdm
from codes.utils import resize, makedirpath, save_pickle, load_pickle
import time
import os
import numpy as np



def save_maps(maps, options, maps_max):
    N = maps.shape[0]
    images = battery.get_x(mode='test')
    labels, classes = battery.get_label()
    
    for n in tqdm(range(N)):
        fig, axes = plt.subplots(ncols=2)
        fig.set_size_inches(6, 3)

        image = resize(images[n], (128, 128))
        image_class = classes[n]

        axes[0].imshow(image)
        axes[0].set_axis_off()

        axes[1].imshow(maps[n], vmax=maps_max, cmap='Reds') # 절대값 기준 => 정상인 애들이 다 하얗게 나온다.
#         axes[1].imshow(maps[n], vmax=maps[n].max(), cmap='Reds') # 상대값 기준 => anomaly인 경우 그림이 이쁨.
        axes[1].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/{options}/n{n:04d}_{image_class}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()

#########################


def main():
    from codes.inspection import eval_encoder_NN_multiK

#     results = eval_encoder_NN_multiK()
    
    #det_64 = results['det_64']

    #det_32 = results['det_32']

    #det_sum = results['det_sum']
#     import pdb;pdb.set_trace()
    load_results = True
    if load_results:
        results = load_pickle('results.pickle')
        print("load results finished...")
    else:
        results = eval_encoder_NN_multiK()
        
        
    det_mult = results['det_mult']

    print(
        f'| mult | Det: {det_mult:.3f}')

    
    maps = results['maps_mult']
    
    save_maps(maps, "mult_absolute", maps.max()) 
    
    #np.save(os.path.join(os.getcwd(), 'maps_mult'), maps4)


if __name__ == '__main__':
    main()

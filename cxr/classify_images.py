import sys, os

import torch
import json
from os.path import join, dirname
from tqdm import tqdm
from PIL import Image
import imageio

from train_mortality_xray import dict_to_dataset

MIN_RES = 256

def main(args):
    if len(args) < 2:
        sys.stderr.write("Reqiured argument(s): <model file> <data file> <mimic path>\n")
        sys.exit(-1)
    
    model = torch.load(args[0])
    infer_file = args[1]
    mimic_path = args[2]

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'

    model.zero_grad()
    model = model.to(device)

    if infer_file.endswith('.json'):
        with open(args[1], 'rt') as fp:
            infer_json = json.load(fp)
            infer_dataset = dict_to_dataset(infer_json, args[1], max_size=-1)

    for inst in tqdm(infer_dataset['data']):
        img_path = join(mimic_path, 'files', inst['images'][-1]['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
        label = inst['out_hospital_mortality_30']
        id = inst['id']
        image = imageio.imread(img_path, mode='F')

if __name__ == '__main__':
    main(sys.argv[1:])
import sys, os

import torch
import torchvision
import json
from os.path import join, dirname
from tqdm import tqdm
from PIL import Image
import imageio
import numpy as np

from train_mortality_xray import dict_to_dataset, cxr_infer_transforms

MIN_RES = 512

def main(args):
    if len(args) < 2:
        sys.stderr.write("Reqiured argument(s): <model file> <data json file> <data pt file>\n")
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
            # infer_dataset = dict_to_dataset(infer_json, args[3], max_size=-1)
    infer_dataset = torch.load(args[2])
    positive = negative=0
    for inst in tqdm(infer_json['data']):
        if 'images' in inst.keys() and len(inst['images']) > 0:
            label = inst['out_hospital_mortality_30']
            if label == 1:
                positive += 1
            elif label == 0:
                negative += 1
            else:
                raise Exception("Unexpected label value %s" % (str(label)))
    
    size = positive + negative
    prevalence = float(positive) / float(size)
    print("In json dataset found %d positive labels in %d instances for a prevalence of %f" % (positive, size, prevalence))

    positive = negative = 0
    for inst in infer_dataset:
        label = inst[1]
        if label == 1:
            positive += 1
        elif label == 0:
            negative += 1
        else:
            raise Exception("Unexepcted label value %s" % (str(label)))
    
    size = positive + negative
    prevalence = float(positive) / float(size)
    print("In cached dataset found %d positive labels in %d instances for a prevalence of %f" % (positive, size, prevalence))

    inst_ind = -1
    for inst in tqdm(infer_json['data']):
        if 'images' in inst.keys() and len(inst['images']) > 0:
            inst_ind += 1
            # img_path = join(mimic_path, 'files', inst['images'][-1]['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
            label = inst['out_hospital_mortality_30']
            inst_id = inst['id']
            inst_data, inst_label = infer_dataset[inst_ind]
            assert label==inst_label
            # image = imageio.imread(img_path, mode='F')
            # inst = cxr_infer_transforms(image)
            if type(model) == torchvision.models.resnet.ResNet:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0,0] = inst_data
                padded_matrix[0,1] = inst_data
                padded_matrix[0,2] = inst_data
                inst_data = padded_matrix.to(device)

            logits = model(inst_data).cpu().detach()
            
            probs = torch.softmax(logits, axis=1)
            pred = torch.argmax(logits, axis=1)

            if pred == 1:
                if pred==label:
                    print("Instance id: %s at path %s was correctly labeled True with probability %f" % (inst_id, inst['images'][-1]['path'], float(probs[0][1])) )
                else:
                    print("Instance id: %s at path %s was INCORRECTLY labeled True with probability %f" % (inst_id, inst['images'][-1]['path'], float(probs[0][1])) )

if __name__ == '__main__':
    main(sys.argv[1:])
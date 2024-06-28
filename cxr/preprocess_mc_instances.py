import sys
import torch
import json
from os.path import join, exists
import logging

from tqdm import tqdm
import imageio
import h5py
import torchvision.transforms as tfms
from PIL import Image

from train_mortality_xray import MIN_RES, cxr_train_transforms, cxr_infer_transforms

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <json file path> <mimic cxr jpg path> <output filename>\n")
        sys.exit(-1)

    input_json, mimic_path, output_fn = args
    if 'train' in input_json:
        print("Using training transforms since the string 'train' was found in the input file")
        transform = cxr_train_transforms
    else:
        print("Using inference transforms since the string 'train' was not found in the input filename.")
        transform = cxr_infer_transforms
        
    with open(input_json, 'rt') as fp:
        data_json = json.load(fp)

    num_insts = 0
    for inst in data_json['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    # need a dataset type that allows for different instances to have different numbers of images.
    ## create a list of lists of image paths for the dataset
    data_paths = []
    labels = {}
    print("Processing data paths")
    for inst in tqdm(data_json['data']):
        if 'images' not in inst or len(inst['images']) == 0:
            continue
        
        adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
        inst_paths = []
        sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
        for image in sorted_images:
            if str(image["StudyDate"]) >= adm_dt:
                img_path = join(mimic_path, 'files', image['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
                inst_paths.append(img_path)
        
        if len(inst_paths) > 0 and len(inst_paths) < 8:
            data_paths.append( (inst['id'], inst_paths) )
            labels[inst['id']] = inst['out_hospital_mortality_30']

    print("Out of %d instances with images, %d have images from the same encounter" % (num_insts, len(data_paths)))

    print("Compiling multi-channel images for each instance")
    with h5py.File(output_fn, "w") as of:
        of['/len'] = len(data_paths)
        for inst_ind, inst in enumerate(tqdm(data_paths)):
            inst_id, inst_image_paths = inst
            inst_images = [imageio.imread(img_path, mode='F') for img_path in inst_image_paths]
            inst_images = [transform(image) for image in inst_images]
            # instances.append(inst_images)
            inst_data_path = '/%d/data' % (inst_ind)
            inst_label_path = '/%d/label' % (inst_ind)
            inst_ind_path = '/%d/id' % (inst_ind)

            of[inst_data_path] = torch.stack(inst_images)
            of[inst_label_path] = labels[inst_id]
            of[inst_ind_path] = inst_ind

if __name__ == '__main__':
    main(sys.argv[1:])

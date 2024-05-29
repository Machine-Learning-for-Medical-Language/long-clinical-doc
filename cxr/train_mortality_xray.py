#!/usr/bin/env python3
#
#####################
# baseline for training from chest xrays
# borrows a lot of image processing stuff from pytorch-cxr: https://github.com/jinserk/pytorch-cxr
#####################

import sys
import os
import json
from os.path import join, exists, dirname
import logging

from tqdm import tqdm

from models.BaselineModel import BaselineMortalityPredictor
from models.MultiChannelModel import MultiChannelMortalityPredictor

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torchvision.transforms as tfms
import torchvision.models as models
import torchxrayvision as xrv

from PIL import Image
import imageio
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


MIN_RES = 256
MEAN = 0.485
STDEV = 0.229
use_resnet = False
# chexnet not working
use_chexnet = False
# multi-channel model
use_mc = False

# From pytorch-cxr
cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN_RES, MIN_RES), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.Resampling.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

cxr_infer_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.CenterCrop(size=(MIN_RES,MIN_RES)),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

def dict_to_dataset(mimic_dict, mimic_path, max_size=-1, train=True, image_selection='last'):
    num_insts = 0
    for inst in mimic_dict['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    print("Found %d instances in this data" % (num_insts))
    if max_size >= 0 and num_insts > max_size:
        num_insts = max_size
        print("Using %d instances due to passed in argument" % (num_insts))

    if image_selection == 'last':
        padded_matrix = torch.zeros(num_insts, MIN_RES, MIN_RES)
        labels = []
        for inst in tqdm(mimic_dict['data']):
            if 'images' not in inst or len(inst['images']) == 0:
                # some instances to not have any xrays
                continue
            # -1 here means that we grab the last image in the list of images, i.e. the latest image. future work shoudl try to use all images.
            # Then we convert it to the resized version, this should save time during loading
            img_path = join(mimic_path, 'files', inst['images'][-1]['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
            if not exists(img_path):
                # this should only happen if we are still processing the images.
                #print("Skipping file %f due to missing image" % (img_path))
                continue
            image = imageio.imread(img_path, mode='F')
            padded_matrix[len(labels), :, :] = cxr_train_transforms(image) if train else cxr_infer_transforms(image)
            labels.append(inst['out_hospital_mortality_30'])
            if len(labels) >= num_insts:
                break

        if len(labels) < num_insts:
            if max_size >= 0:
                raise Exception("Not enough data points in this dataset to satisfy the requested number of instances! %d vs. %d" % (len(labels), num_insts))
            else:
                logging.warning("Missing some image files so actual dataset is smaller than number of instances: %d vs. %d. Truncating input feature matrix" % (len(labels), num_insts))
                padded_matrix = padded_matrix[:len(labels)]

        dataset = TensorDataset(padded_matrix, torch.tensor(labels))
    elif image_selection == 'all':
        # need a dataset type that allows for different instances to have different numbers of images.
        dataset = None
        raise NotImplementedError()

    return dataset

def run_one_eval(model, eval_dataset, device, loss_fct):
    num_correct = num_wrong = 0
    model.eval()

    preds = np.zeros(len(eval_dataset))
    test_labels = np.zeros(len(eval_dataset))

    with torch.no_grad():
        dev_loss = 0
        for ind in range(0, len(eval_dataset)):
            matrix, label = eval_dataset[ind]
            # label_ind = label_map[label]
            test_labels[ind] = label
            if use_resnet:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0,0] = matrix
                padded_matrix[0,1] = matrix
                padded_matrix[0,2] = matrix
            else: # chexnet can use the default
                padded_matrix = torch.zeros(1, MIN_RES, MIN_RES)
                padded_matrix[0] = matrix

            logits = model(padded_matrix.to(device))
            loss = loss_fct(logits[0], label.to(device))
            dev_loss = loss.item()
            pred = np.argmax(logits.cpu().numpy(), axis=1)
            preds[ind] = pred

            if pred == label:
                num_correct += 1
            else:
                num_wrong += 1
        print("Dev loss is %f" % (dev_loss))

    # accuracy = num_correct / len(dev_dataset)
    acc = accuracy_score(test_labels, preds)
    #print("Final accuracy on held out data was %f" % (acc))
    f1 = f1_score(test_labels, preds, average=None)
    rec = recall_score(test_labels, preds, average=None)
    prec = precision_score(test_labels, preds, average=None)
    #print("F1 score is %s" % (str(f1)))
    return {'dev_loss': dev_loss, 'acc': acc, 'f1': f1, 'rec': rec, 'prec': prec}

def main(args):
    if len(args) < 4:
        sys.stderr.write('Required arguments: <train file> <dev file> <mimic-cxr-jpg root> <filename for saved model>\n')
        sys.exit(-1)
 
    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'

    random_seed = 42
    num_epochs = 200
    batch_size = 64
    eval_freq = 10
    # 0.001 is Adam default
    learning_rate=0.0001

    if use_resnet:
        model = models.resnet18(pretrained=True)
        # need to modify the output layer for our label set (plus we don't care about their output space so those weights are not interesting to us)
        # 512x2 for resnet18, 1024x2 for chex
        model.fc = nn.Linear(512, 2)
    elif use_chexnet:
        model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        model.fc = nn.Linear(1024, 2)
    elif use_mc:
        model = MultiChannelMortalityPredictor( (MIN_RES, MIN_RES) )
    else:
        model = BaselineMortalityPredictor( (MIN_RES, MIN_RES) )

    model.zero_grad()
    model = model.to(device)
    loss_fct = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if args[0].endswith('.json'):
        with open(args[0], 'rt') as fp:
            train_json = json.load(fp)
            if use_mc:
                train_dataset = dict_to_dataset(train_json, args[2], max_size=-1, train=True, image_selection='all')
            else:
                train_dataset = dict_to_dataset(train_json, args[2], max_size=-1, train=True)
            out_file = join(dirname(args[1]), 'train_cache_res=%d_selection=%s.pt' % (MIN_RES, 'all' if use_mc else 'last'))
            torch.save(train_dataset, out_file)
    else:
        print("Loading training data from cache.")
        train_dataset = torch.load(args[0])
        
    if args[1].endswith('.json'):    
        with open(args[1], 'rt') as fp:
            dev_json = json.load(fp)
            if use_mc:
                dev_dataset = dict_to_dataset(dev_json, args[2], max_size=-1, train=False, image_selection='all')
            else:
                dev_dataset = dict_to_dataset(dev_json, args[2], max_size=-1, train=False)

            out_file = join(dirname(args[1]), 'dev_cache_res=%d_selection=%s.pt' % (MIN_RES, 'all' if use_mc else 'last'))
            torch.save(dev_dataset, out_file)
    else:
        print("Loading dev data from cache.")
        dev_dataset = torch.load(args[1])

    save_fn = args[3]
    
    sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    best_loss = -1
    all_dev_results = {}
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        #for count,ind in enumerate(train_inds):
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            batch_data, batch_labels = batch
            opt.zero_grad()
            if use_resnet:
                # resnet expects 3 channels (RGB) but we have grayscale images
                rgb_batch = torch.repeat_interleave(batch_data.unsqueeze(1), 3, dim=1)
                logits = model(rgb_batch)
            elif use_chexnet:
                # Not working
                logits = model(batch_data.unsqueeze(1).to(device)*341)
            else:  # chexnet uses the deafult
                # add a empty 1 dimension here for the 1 channel so the model knows
                # the first dimension is batch and not channels
                logits = model(batch_data.unsqueeze(1).to(device))
            loss = loss_fct(logits, batch_labels.to(device))
            loss.backward()
            epoch_loss += loss.item()
            opt.step()
            model.zero_grad()
        print("Epoch %d loss: %0.9f" % (epoch, epoch_loss))
        if epoch % eval_freq == 0:
            dev_results = run_one_eval(model, dev_dataset, device, loss_fct)
            for result_key, result_val in dev_results.items():
                print("Dev %s = %s" % (result_key, str(result_val)))
                if result_key not in all_dev_results:
                    all_dev_results[result_key] = []
                all_dev_results[result_key].append(result_val)

        if best_loss < 0 or epoch_loss < best_loss:
            best_loss = epoch_loss
            print("Saving model")
            torch.save(model, save_fn)
    #torch.save(model, save_fn)

    # load the best model rather than the most recent
    model = torch.load(save_fn)
    final_dev_results = run_one_eval(model, dev_dataset, device, loss_fct)
    print("Final dev results: %s" % str(final_dev_results))

    print("Running dev results:")
    print(str(all_dev_results))

 
if __name__ == '__main__':
    main(sys.argv[1:])
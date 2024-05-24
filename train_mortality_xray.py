#!/usr/bin/env python3
#
#####################
# baseline for training from chest xrays
# borrows a lot of image processing stuff from pytorch-cxr: https://github.com/jinserk/pytorch-cxr
#####################

import sys
import os
import json
from os.path import join, exists

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as tfms
from PIL import Image
import imageio

MIN_RES = 256
MEAN = 0.4
STDEV = 0.2

class BaselineMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=32, kernel_size=5, pool_size=10):

        super(BaselineMortalityPredictor, self).__init__()
        # in = embed_dim, out = num_filters, kernel = 5
        # default stride=1, padding=0, dilation=1
        stride = 1
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size)
        self.hout_conv = int(torch.floor( torch.tensor(1 + (shape[0] - (kernel_size-1) - 1) / stride)).item())
        self.wout_conv = int(torch.floor( torch.tensor(1 + (shape[1] - (kernel_size-1) - 1) / stride)).item())
        
        self.pool = nn.MaxPool2d(pool_size)
        pool_stride = pool_size
        self.hout_pool = int(torch.floor( torch.tensor(1 + (self.hout_conv - (pool_size-1) - 1) / pool_stride)).item())
        self.wout_pool = int(torch.floor( torch.tensor(1 + (self.wout_conv - (pool_size-1) - 1) / pool_stride)).item())
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(num_filters * self.hout_pool * self.wout_pool, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        print("Model initialized with hidden layer containing %d nodes" % (self.fc1.in_features) )
    
    def forward(self, matrix):
        #unpooled = F.relu(self.conv1(embedded.permute(0,3,1,2)))
        unpooled = F.relu(self.conv1(matrix))
        pooled = self.pool(unpooled)
        x = pooled.view(matrix.shape[0], -1)
        # pooled = self.pool(F.relu(self.conv1(embedded)))
        # pooled = self.pooling(self.conv2d_3(embedded))
        out = self.fc1(x)
        return out

# From pytorch-cxr
cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN_RES, MIN_RES), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

def dict_to_dataset(mimic_dict, mimic_path, max_size=-1):
    num_insts = 0
    for inst in mimic_dict['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    print("Found %d instances in this data" % (num_insts))
    if max_size >= 0 and num_insts > max_size:
        num_insts = max_size
        print("Using %d instances due to passed in argument" % (num_insts))

    padded_matrix = torch.zeros(num_insts, MIN_RES, MIN_RES)
    labels = []
    for inst_num,inst in tqdm(enumerate(mimic_dict['data'])):
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
        padded_matrix[len(labels), :, :] = cxr_train_transforms(image)
        labels.append(inst['mortality'])
        if len(labels) >= num_insts:
            break

    if len(labels) < num_insts:
        raise Exception("Not enough data points in this dataset to satisfy the requested number of instances! %d vs. %d" % (len(labels), num_insts))
    dataset = TensorDataset(padded_matrix, torch.tensor(labels))

    return dataset

def main(args):
    if len(args) < 3:
        sys.stderr.write('Required arguments: <train file> <dev file> <mimic-cxr-jpg root> <filename for saved model>\n')
        sys.exit(-1)

    with open(args[0], 'rt') as fp:
        train_json = json.load(fp)
    
    with open(args[1], 'rt') as fp:
        dev_json = json.load(fp)
    
    save_fn = args[3]

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'
 
    train_dataset = dict_to_dataset(train_json, args[2], max_size=100)
    dev_dataset = dict_to_dataset(dev_json, args[2], max_size=100)

    print("Done reading data and quitting!")

    random_seed = 42
    num_epochs = 10
    batch_size = 8

    sampler = RandomSampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    model = BaselineMortalityPredictor( (MIN_RES, MIN_RES) ) 
    model.zero_grad()
    model = model.to(device)
    loss_fct = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    training_loss = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        #for count,ind in enumerate(train_inds):
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            batch_data, batch_labels = batch
            # add a empty 1 dimension here for the 1 channel so the model knows
            # the first dimension is batch and not channels
            logits = model(batch_data.unsqueeze(1).to(device))
            loss = loss_fct(logits, batch_labels.to(device))
            loss.backward()
            epoch_loss += loss.item()
            opt.step()
            model.zero_grad()
        print("Epoch %d loss: %0.9f" % (epoch, epoch_loss))
    torch.save(model, save_fn)


if __name__ == '__main__':
    main(sys.argv[1:])
#!/usr/bin/env python3
#
#####################
# baseline for training from chest xrays
# borrows a lot of image processing stuff from pytorch-cxr: https://github.com/jinserk/pytorch-cxr
#####################

import sys
import json
from os.path import join, exists, dirname
from dataclasses import dataclass, field

from tqdm import tqdm

from models.BaselineModel import BaselineMortalityPredictor
from models.MultiChannelModel import MultiChannelMortalityPredictor, VariableLengthImageDataset, collate_fn
from models.MultiModalModel import MultiModalMortalityPredictor
# from models.VitModel import VitMortalityPredictor

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch.nn.functional as F
import torchvision.transforms as tfms
import torchvision.models as models
import torchxrayvision as xrv
from transformers import HfArgumentParser
from PIL import Image
import imageio
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from scipy.special import softmax

RESNET = 'resnet'
BASELINE = "baseline"
MULTICHANNEL = "mc"
CHEXNET = "chexnet"
VIT = "vit"
MULTIMODAL = "multimodal"

# From pytorch-cxr
MIN_RES = 512
MEAN = 0.0
STDEV = 0.229

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN_RES, MIN_RES), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.Resampling.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

cxr_infer_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((MIN_RES,MIN_RES), interpolation=Image.Resampling.LANCZOS),
    tfms.CenterCrop(MIN_RES),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

def dict_to_dataset(mimic_dict, mimic_path, max_size=-1, train=True, image_selection='last'):
    num_insts = 0
    for inst in mimic_dict['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    metadata = []
    print("Found %d instances in this data" % (num_insts))
    if max_size >= 0 and num_insts > max_size:
        num_insts = max_size
        print("Using %d instances due to passed in argument" % (num_insts))

    if image_selection == 'last':
        insts = []
        labels = []
        for inst in tqdm(mimic_dict['data']):
            inst_metadata = {}
            if 'images' not in inst or len(inst['images']) == 0:
                # some instances to not have any xrays
                continue
            # -1 here means that we grab the last image in the list of images, i.e. the latest image. future work shoudl try to use all images.
            # Then we convert it to the resized version, this should save time during loading
            # Sort so they are in order
            adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
            sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
            ## navigate backwards in time through images and try to grab the right kind of image
            for image in sorted_images:
                # Has to be part of this admission and be a portable chest xray
                # we already check before we build this .json file that it's before the discharge date
                if str(image["StudyDate"]) >= adm_dt and image["PerformedProcedureStepDescription"] == "CHEST (PORTABLE AP)":
                    img_path = join(mimic_path, 'files', image['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
                    if not exists(img_path):
                        # this should only happen if we are still processing the images.
                        #print("Skipping file %f due to missing image" % (img_path))
                        raise Exception("Path to image does not exist: %s" % (img_path))
                    img_data = imageio.imread(img_path, mode='F')
                    insts.append(cxr_train_transforms(img_data) if train else cxr_infer_transforms(img_data))
                    labels.append(inst['out_hospital_mortality_30'])
                    inst_metadata['img_fn'] = img_path
                    inst_metadata['hadm_id'] = inst["debug_features"]["HADM_ID"]
                    metadata.append(inst_metadata)
                    break

            if len(labels) >= num_insts:
                break

        padded_matrix = torch.stack(insts).squeeze()
        dataset = TensorDataset(padded_matrix, torch.tensor(labels))
    else:
        raise NotImplementedError("Image selection method %s is not implemented" % (image_selection))
        
    return dataset, metadata

def run_one_eval(model, eval_dataset, device, model_type):
    num_correct = num_wrong = 0
    model.eval()

    preds = np.zeros(len(eval_dataset))
    test_labels = np.zeros(len(eval_dataset))
    test_probs = []

    with torch.no_grad():
        dev_loss = 0
        for ind in range(0, len(eval_dataset)):
            if len(eval_dataset[ind]) == 3:
                matrix, text, label = eval_dataset[ind]
            else:
                matrix, label = eval_dataset[ind]

            # label_ind = label_map[label]
            test_labels[ind] = label
            if model_type==RESNET or model_type==VIT:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0,0] = matrix
                padded_matrix[0,1] = matrix
                padded_matrix[0,2] = matrix
                logits = model(padded_matrix.to(device))
            elif model_type==MULTICHANNEL or model_type == MULTIMODAL:
                # train shape is (batch_size, 1, num_images, res, res) - 1 is grayscale channel
                # shape from reader is (num_images, 1, res, res)
                # we want (for evals of batch size 1) shape (1, 1, num_images, 512, 512)
                padded_matrix = torch.permute(matrix, (1,0,2,3)).unsqueeze(dim=0)
                if model_type == MULTICHANNEL:
                    logits = model(padded_matrix.to(device))
                elif model_type == MULTIMODAL:
                    logits = model(img_matrix=padded_matrix.to(device), text=[text])
                else:
                    raise NotImplementedError("Only multichannel or multimodal models are handled here.")
            else: # chexnet can use the default
                padded_matrix = matrix.unsqueeze(dim=0)

                logits = model(padded_matrix.to(device))

            label = torch.tensor(label)
            loss = F.cross_entropy(logits, label.unsqueeze(dim=0).to(device))
            dev_loss = loss.item()
            pred = np.argmax(logits.cpu().numpy(), axis=1)
            preds[ind] = pred
            test_probs.append(torch.softmax(logits.cpu(), dim=1)[:,1].numpy())

            if pred == label:
                num_correct += 1
            else:
                num_wrong += 1
        #print("Dev loss is %f" % (dev_loss))

    # accuracy = num_correct / len(dev_dataset)
    acc = accuracy_score(test_labels, preds)
    #print("Final accuracy on held out data was %f" % (acc))
    f1 = f1_score(test_labels, preds, average=None)
    rec = recall_score(test_labels, preds, average=None)
    prec = precision_score(test_labels, preds, average=None)
    prev = test_labels.sum() / len(test_labels)
    auroc = roc_auc_score(y_true=test_labels, y_score=test_probs)
    precs, recs, thresholds = precision_recall_curve(y_true=test_labels, y_score=test_probs, drop_intermediate=True)
    max_f1 = -1
    max_threshold = 0.0
    max_f1_prec = max_f1_rec = 0

    for dp_ind, threshold in enumerate(thresholds):
        dp_prec = precs[dp_ind]
        dp_rec = recs[dp_ind]
        dp_f1 = 2 * dp_prec * dp_rec / (dp_prec+dp_rec)
        if dp_f1 > max_f1:
            max_f1 = dp_f1
            max_threshold = threshold
            max_f1_rec = dp_rec
            max_f1_prec = dp_prec

    #print("F1 score is %s" % (str(f1)))
    return {'dev_loss': dev_loss, 'acc': acc, 'f1': f1, 'rec': rec, 'prec': prec, 'prevalence': prev, 'auroc': auroc, 'prc': {'prec':max_f1_prec, 'rec':max_f1_rec, 'f1':max_f1, 'thresh':max_threshold}}

@dataclass
class TrainingArguments:
    train_file: str = field(
        metadata={"help": ".json, .pt, or .hdf5 file for training"}
    )
    eval_file: str = field(
        metadata={"help": ".json, .pt, or .hdf5 file for evaluating during training"}
    )
    cxr_root: str = field(
        default=None,
        metadata={"help": "Path to mimic-cxr-jpg data (root of the dir for a specific version)"}
    )
    num_training_epochs: int = field(
        default=10,
        metadata={"help": "Number of passes to take over the training dataset"}
    )
    model_filename: str = field(
        default=None,
        metadata={"help": "Filename to write saved .pt model."}
    )
    seed: int = field(
        default=42,
        metadata={"help":"Random seed to use during training (currently has no effect)"}
    )
    batch_size: int = field(
        default=10,
        metadata={"help":"Number of instances to include in each training batch"}
    )
    eval_freq: int = field(
        default=10,
        metadata={"help":"How often do evals in terms of number of training epochs"}
    )
    learning_rate: float = field(
        default=0.001,
        metadata={"help":"Learning rate to pass to optimizer"}
    )
    model: str = field(
        default='baseline',
        metadata={"choices": [RESNET, MULTICHANNEL, BASELINE, CHEXNET, VIT, MULTIMODAL],
                  "help": "Which neural model to use"}
    )
    label_field: str = field(
        default="out_hospital_mortality_30",
        metadata={"help": "The field to grab from the dataset file to use as the label"}
    )
    max_train: int = field(
        default=-1,
        metadata={"help":"The maximum number of instances to use for training (for quick testing)"}
    )
    max_eval: int = field(
        default=-1,
        metadata={"help":"The maximum number of instances to use for evaluations (for quick testing)"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "The random seed to pass to torch.cuda.manual_seed() when the program is started. Probably affects neural net initialization and batch shuffling."}
    )
    img_model: str = field(
        default=None,
        metadata={"help": "The saved trained model to use to initialize the image classification model. Used in the multi-modal classifier."}
    )
    txt_model: str = field(
        default=None,
        metadata={"help": "The saved trained model to use to initialize a text classification model. Used in the multi-modal classifier."}
    )
    ga: int = field(
        default=1,
        metadata={"help": "Gradient accumulation -- approximate larger batch sizes by skipping gradient clears for ga steps. Effective batch size is ga * batch_size."}
    )
     
def main(args): 
    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        sys.err.print("CUDA was not found available.... backing off to CPU, this may be very slow...")
        sys.exit(-1)
        device = 'cpu'

    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    batch_size = training_args.batch_size
    eval_freq = training_args.eval_freq
    ga = training_args.ga
    
    print("Training args: %s" % (str(training_args)))

    torch.cuda.manual_seed(training_args.seed)
    # 0.001 is Adam default
    learning_rate=training_args.learning_rate

    if training_args.model == RESNET:
        print("Using resnet model")
        model = models.resnet34(pretrained=True)
        # need to modify the output layer for our label set (plus we don't care about their output space so those weights are not interesting to us)
        # 512x2 for resnet18, 1024x2 for chex
        model.fc = nn.Linear(512, 2)
    elif training_args.model == CHEXNET:
        print("Using chexnet: this isn't working right now")
        raise NotImplementedError("Chexnet is not working now.")
        model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        model.fc = nn.Linear(1024, 2)
    elif training_args.model == MULTICHANNEL:
        print("Using multi-channel model")
        model = MultiChannelMortalityPredictor( (MIN_RES, MIN_RES) )
        if training_args.eval_file.endswith('.json'):
            import bmemcached
    elif training_args.model == VIT:
        model = models.vision_transformer.vit_l_16(models.vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.heads.head = nn.Linear(1024, 2)
    elif training_args.model == MULTIMODAL:
        model = MultiModalMortalityPredictor(img_model=training_args.img_model, text_model=training_args.txt_model)
    else:
        print("Using baseline single image (CNN) model")
        model = BaselineMortalityPredictor( (MIN_RES, MIN_RES) )

    model.zero_grad()
    model = model.to(device)
    loss_fct = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Loading training data...")
    if training_args.train_file.endswith('.json'):
        with open(training_args.train_file, 'rt') as fp:
            train_json = json.load(fp)
            if training_args.model==MULTICHANNEL:
                raise Exception("For multi-channel models, dataset must be pre-processed first!")
            else:
                train_dataset, _ = dict_to_dataset(train_json, training_args.cxr_root, max_size=training_args.max_train, train=True)
            out_file = join(dirname(training_args.train_file), 'train_cache_res=%d_selection=%s.pt' % (MIN_RES, 'all' if training_args.model==MULTICHANNEL else 'last'))
            torch.save(train_dataset, out_file)
    elif training_args.train_file.endswith('.pt'):
        print("Loading training data from cached pytorch file.")
        train_dataset = torch.load(training_args.train_file)
    elif training_args.train_file.endswith('.hdf5'):
        train_dataset = VariableLengthImageDataset(training_args.train_file)

    print("Loading evaluation data...")
    if training_args.eval_file.endswith('.json'):    
        with open(training_args.eval_file, 'rt') as fp:
            dev_json = json.load(fp)
            if training_args.model == MULTICHANNEL:
                raise Exception("For multi-channel models, dataset must be pre-processed first!")
            else:
                dev_dataset, metadata = dict_to_dataset(dev_json, training_args.cxr_root, max_size=training_args.max_eval, train=False)
            out_file = join(dirname(training_args.eval_file), 'dev_cache_res=%d_selection=%s.pt' % (MIN_RES, 'all' if training_args.model == MULTICHANNEL else 'last'))
            torch.save(dev_dataset, out_file)
    elif training_args.eval_file.endswith('.pt'):
        print("Loading dev data from cache.")
        dev_dataset = torch.load(training_args.eval_file)
    elif training_args.eval_file.endswith('.hdf5'):
        dev_dataset = VariableLengthImageDataset(training_args.eval_file)
 
    sampler = RandomSampler(train_dataset)
    if training_args.model == MULTICHANNEL or training_args.model == MULTIMODAL:
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    else:
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    best_loss = None
    all_dev_results = {}
    for epoch in range(training_args.num_training_epochs):
        model.train()
        epoch_loss = 0.0
        accum_steps = 0
        #for count,ind in enumerate(train_inds):
        for batch in tqdm(dataloader, desc="Iteration"):
            if len(batch) == 3:
                batch_images, batch_text, batch_labels = batch
            elif len(batch) == 2:
                batch_images, batch_labels = batch

            batch_images = torch.stack(tuple(t.to(device) for t in batch_images))
            batch_labels = torch.stack(tuple(t.to(device) for t in batch_labels))
            
            if training_args.model == RESNET:
                # resnet expects 3 channels (RGB) but we have grayscale images
                rgb_batch = torch.repeat_interleave(batch_images.unsqueeze(1), 3, dim=1)
                logits = model(rgb_batch)
            elif training_args.model == CHEXNET:
                # Not working
                logits = model(batch_images.unsqueeze(1)*341)
            elif training_args.model == VIT:
                rgb_batch = torch.repeat_interleave(batch_images.unsqueeze(1), 3, dim=1)
                logits = model(rgb_batch)
            elif training_args.model == MULTICHANNEL or training_args.model == MULTIMODAL:
                # add a empty 1 dimension here for the 1 channel so the model knows
                # the first dimension is batch and not channels
                if training_args.model == MULTICHANNEL:
                    logits = model(batch_images.unsqueeze(1))
                elif training_args.model == MULTIMODAL:
                    logits = model(img_matrix=batch_images.unsqueeze(1), text=batch_text)

            loss = loss_fct(logits, batch_labels)
            loss.backward()
            accum_steps += 1

            if accum_steps % ga == 0:
                opt.step()
                opt.zero_grad()
                epoch_loss += loss.item()
                accum_steps = 0

        print("Epoch %d loss: %0.9f" % (epoch, epoch_loss))
        if epoch % eval_freq == 0:
            dev_results = run_one_eval(model, dev_dataset, device, training_args.model)
            dev_loss = dev_results['dev_loss']
            dev_auroc = dev_results['auroc']
            thing_to_minimize=-dev_auroc # save when auroc is better than before
            for result_key, result_val in dev_results.items():
                print("Dev %s = %s" % (result_key, str(result_val)))
                if result_key not in all_dev_results:
                    all_dev_results[result_key] = []
                all_dev_results[result_key].append(result_val)

            if not best_loss or thing_to_minimize < best_loss:
                best_loss = thing_to_minimize
                if training_args.model_filename:
                    print("Saving model")
                    torch.save(model, training_args.model_filename)

    # load the best model rather than the most recent
    if training_args.model_filename:
        torch.load(training_args.model_filename)

    final_dev_results = run_one_eval(model, dev_dataset, device, training_args.model)
    print("Final dev results: %s" % str(final_dev_results))

    print("Running dev results:")
    print(str(all_dev_results))

 
if __name__ == '__main__':
    main(sys.argv[1:])

import sys

import json
import tqdm

from dataclasses import dataclass, field
import numpy as np
from transformers import HfArgumentParser
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from train_mortality_xray import dict_to_dataset, RESNET, MULTICHANNEL, BASELINE, CHEXNET, MIN_RES

@dataclass
class InferenceArguments:
    eval_file: str = field(
        metadata={"help": ".json, .pt, or .hdf5 file for evaluating during training"}
    )
    cxr_root: str = field(
        metadata={"help": "Path to mimic-cxr-jpg data (root of the dir for a specific version)"}
    )
    model_file: str = field(
        metadata={"help": "Pytorch model file to load to do the processing"}
    )
    model_type: str = field(
        default='baseline',
        metadata={"choices": [RESNET, MULTICHANNEL, BASELINE, CHEXNET],
                  "help": "Which neural model to use"}
    )
    max_eval: int = field(
        default=-1,
        metadata={"help":"The maximum number of instances to use for evaluations (for quick testing)"}
    )


def main(args):
    
    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'

    parser = HfArgumentParser(InferenceArguments)
    infer_args, = parser.parse_args_into_dataclasses()
    
    model = torch.load(infer_args.model_file)

    with open(infer_args.eval_file, 'rt') as fp:
        dev_json = json.load(fp)
        if infer_args.model_type == MULTICHANNEL:
            raise Exception("For multi-channel models, dataset must be pre-processed first!")
        else:
            eval_dataset, metadata = dict_to_dataset(dev_json, infer_args.cxr_root, max_size=infer_args.max_eval, train=False)

    model.eval()
    print("True label\tProb(death)\tPredicted label\tHADM_ID\tPath")
    with torch.no_grad():
        for ind in range(0, len(eval_dataset)):
            matrix, label = eval_dataset[ind]
            if infer_args.model_type==RESNET:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0,0] = matrix
                padded_matrix[0,1] = matrix
                padded_matrix[0,2] = matrix
            elif infer_args.model_type==MULTICHANNEL:
                # train shape is (batch_size, 1, num_images, res, res)
                # shape from reader is (num_images, 1, res, res)
                # we want (for evals of batch size 1) shape (1, 1, num_images, 512, 512)
                padded_matrix = torch.permute(matrix, (1,0,2,3)).unsqueeze(dim=0)
            else: # chexnet can use the default
                padded_matrix = torch.zeros(1, MIN_RES, MIN_RES)
                padded_matrix[0] = matrix

            logits = model(padded_matrix.to(device)).cpu()
            probs = torch.softmax(logits, axis=1).numpy()[0]
            pred = np.argmax(logits.numpy(), axis=1)
            fn = metadata[ind]['img_fn']
            hadm_id = metadata[ind]['hadm_id']
            print("%d\t%f\t%d\t%s\t%s" % (label, probs[1], pred[0], hadm_id, fn))
        

if __name__ == '__main__':
    main(sys.argv[1:])

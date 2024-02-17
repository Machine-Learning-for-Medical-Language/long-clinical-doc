import os
import argparse
import hashlib
import pandas as pd
import json
import re
import logging

VERSION = "v20240209"

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
"""
    Set logger level as INFO for per patient-level evaluation results. 
    Set DEBUG for more detailed results.
"""

def hash_data(key, text_seed, seed="Landmark Center 401 park drive", algorithm='sha256', debug=False):
    # Sanity check
    if type(text_seed) != str:
        raise TypeError(f"text_seed should be str type: current: {type(text_seed)}")

    if algorithm=='sha256':
        h_key = hashlib.sha256(key, usedforsecurity=True)
        h_text = hashlib.sha256(text_seed, usedforsecurity=True)
        h_seed = hashlib.sha256(seed, usedforsecurity=True)
    else:
        #hash_func = hashlib.new(algorithm)
        raise NotImplementedError("Hash algorithm must be SHA256!")
    
    xored = h_key ^ h_text ^ h_seed

    if debug:
        logger.debug(f"key:{key}, h_key: {h_key}")
        logger.debug(f"text_seed:{text_seed}, h_text: {h_text}")
        logger.debug(f"seed:{seed}, h_seed: {h_seed}")
        logger.debug(f"XORed:{xored}")

    return xored

def remove_newline(review):
    review = review.replace('&#039;', "'")
    review = review.replace('\n', ' <cr> ')
    review = review.replace('\r', ' <cr> ')
    review = review.replace('\t', ' ')
    review = review.replace('"', '')
    rx = rx = re.compile(r'_{2,}\s*')
    review = rx.sub('', review)
    return " ".join(review.split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Long Clinical Document Benchmark datasets"
    )
    parser.add_argument(
        "--gold_path", required=True, help="A gold annotation json file"
    )
    parser.add_argument(
        "--discharge_path", required=True, help="Path to discharge.csv file"
    )
    parser.add_argument(
        "--output_path", required=True, help="A gold annotation json file"
    )
    args = parser.parse_args()

    logger.info(args)

    gold_data = {}
    for data_type in ["train", "dev", "test"]:
        with open(os.path.join(args.gold_path, f"{data_type}-full.json")) as goldfp:
            gold_data[data_type] = json.load(goldfp)

    outputs_dict = {}

    for data_type, data in gold_data.items():
        for data_idx, data_row in enumerate(data):

            text = data_row['text']

            if data_idx==0:
                debug = True
                logger.debug(f"First sample: ")
            else: 
                debug = False

            note_id = eval(data_row['debug_features']['note_ids'])[0]
            assert type(note_id) == str
            hashed = hash_data(
                key=str(note_id), 
                text_seed=text,
                debug=debug
            )
    
            outputs_dict[hashed] = {
                "data_type": data_type,
                "out_hospital_mortality_30": data_row['out_hospital_mortality_30']
            } 
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if os.path.isdir(args.output_path):
        json_path = os.path.json(args.output_path, "labels.json")
    else:
        json_path = args.output_path
                        
    with open(json_path, 'w') as outfp:
        json.dump(fp=outfp, obj=outputs_dict, indent=2)

    logger.debug(f"Writing of labels done. Path: {args.output_path}")
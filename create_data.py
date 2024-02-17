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
        "--label_path", required=True, help="A gold annotation json file"
    )
    parser.add_argument(
        "--discharge_path", required=True, help="Path to discharge.csv file"
    )
    parser.add_argument(
        "--output_path", required=True, help="A path to save a full dataset (json format)"
    )
    args = parser.parse_args()

    logger.info(args)

    data = pd.read_csv(args.discharge_path)

    with open(args.label_path) as goldfp:
        gold_data = json.load(goldfp)

    outputs_dict = {}
    
    for data_idx, data_row in data.iterrows():
        if data_row['note_id'] not in gold_data:
            continue

        text = remove_newline(data_row['text'])

        if data_idx==0:
            debug = True
            logger.debug(f"First sample: ")
        else: 
            debug = False

        note_id = data_row['note_id']
        assert type(note_id) == str

        hashed = hash_data(
            key=str(note_id), 
            text_seed=text,
            debug=debug
            )
    
        outputs_dict[hashed] = {
            "text": text,
            "out_hospital_mortality_30": gold_data["out_hospital_mortality_30"],
            "data_type": gold_data["data_type"],
        }
    
    for data_type in ["train", "dev", "test"]:
        with open(os.path.join(args.output_path, f"{data_type}.json"), 'w') as outfp:
            json.dump(fp=outfp, obj=outputs_dict, indent=2)

    logger.debug(f"Writing of benchmark dataset done. Path: {args.output_path}")
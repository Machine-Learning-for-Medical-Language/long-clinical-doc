import os, sys
import argparse
import hashlib
import pandas as pd
import json
import re
import logging
from tqdm import tqdm

VERSION = "v20240901"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

normal_data_count = {"train":34759, "dev":7505, "test":7568}

def hash_data(key, text_seed, seed="Landmark Center 401 park drive", algorithm='sha256', debug=False):
    # Sanity check
    if type(text_seed) != str:
        raise TypeError(f"text_seed should be str type: current: {type(text_seed)}")

    if algorithm=='sha256':
        h_key = hashlib.sha256(key.encode('utf-8'), usedforsecurity=True).hexdigest()
        h_text = hashlib.sha256(text_seed.encode('utf-8'), usedforsecurity=True).hexdigest()
        h_seed = hashlib.sha256(seed.encode('utf-8'), usedforsecurity=True).hexdigest()
    else:
        #hash_func = hashlib.new(algorithm)
        raise NotImplementedError("Hash algorithm must be SHA256!")
    
    xored = hex(int(h_key, 16) ^ int(h_text, 16) ^ int(h_seed, 16))

    if debug:
        logger.warning(f"h_key: {h_key}")
        logger.warning(f"key:{key}")
        logger.warning(f"h_text: {h_text}")
        logger.warning(f"text_seed:{text_seed[:300]} (omitted)")
        logger.warning(f"h_seed: {h_seed}")
        logger.warning(f"seed:{seed}")
        logger.warning(f"XORed:{xored}")

    return xored[:32]

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
        "--label_path", required=True, help="A gold annotation json file (label.json)"
    )
    parser.add_argument(
        "--discharge_path", required=True, help="Path to discharge.csv file"
    )
    parser.add_argument(
        "--output_path", required=True, help="A path to save full dataset files (json format)"
    )
    parser.add_argument(
        "--task_name", default="out_hospital_mortality_30", 
        help="Name of the task. Currently supported tasks: out_hospital_mortality_30, out_hospital_mortality_60, out_hospital_mortality_90. Default: out_hospital_mortality_30"
    ) 
    args = parser.parse_args()

    logger.info(args)

    data = pd.read_csv(args.discharge_path)

    with open(args.label_path) as goldfp:
        gold_data = json.load(goldfp)

    outputs_dict_total = {}
    outputs_metadata = {}
    metadata_list = "note_id,subject_id,hadm_id,note_type,note_seq,charttime,storetime".split(",")

    for data_idx, data_row in tqdm(data.iterrows(), total=len(data)):
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
    
        if hashed not in gold_data:
            continue

        outputs_dict_total[hashed] = {
            "text": text,
            str(args.task_name): gold_data[hashed][str(args.task_name)],
            "data_type": gold_data[hashed]["data_type"],
        }
        outputs_metadata[hashed] = {
            metadata_name:data_row[metadata_name] for metadata_name in metadata_list
        }
        outputs_metadata[hashed]["data_type"] = gold_data[hashed]["data_type"]
    
    for data_type in ["train", "dev", "test"]:

        outputs_list = []
        for key, value in outputs_dict_total.items():
            if value['data_type']==data_type:
                value.update({"id":key})
                outputs_list.append(value)
        
        outputs_dict = {
            'data':outputs_list
        }

        with open(os.path.join(args.output_path, f"{data_type}.json"), 'w') as outfp:
            json.dump(fp=outfp, obj=outputs_dict, indent=2)

        logger.warning(f"Writing of {data_type}.json done. # of datapoints: {len(outputs_list)}")
        if len(outputs_list) != normal_data_count[data_type]:
            logger.critical(f"WARNING: The number of datapoints is not matching with the author's processing results!"+\
            "Please contact the authors of the benchmark dataset.")

    with open(os.path.join(args.output_path, "metadata.json"), "w") as outfp:
        json.dump(fp=outfp, obj=outputs_metadata, indent=2)

    logger.debug(f"Writing of benchmark dataset done. Path: {args.output_path}")

import sys
import os
from os.path import join, exists, basename
import json
import glob
import tqdm

import pandas as pd

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <input file> <mimic-cxr-jpg base dir> <output file>\n")
        sys.exit(-1)

    if not exists(args[0]):
        sys.stderr.write("Input file doesn't exist!\n")
        sys.exit(-1)
    
    if not args[0].endswith(".json"):
        sys.stderr.write("Input file should end with .json!\n")
        sys.exit(-1)
    
    if not exists(join(args[1], "files")):
        sys.stderr.write("MIMIC CXR directory does not contain a 'files' directory!\n")
        sys.exit(-1)

    meta_df = pd.read_csv(join(args[1], "mimic-cxr-2.0.0-metadata.csv"))

    with open(args[0], 'rt') as fp:
        json_file = json.load(fp)
        missing_images = 0
        num_insts = 0

        for inst in tqdm.tqdm(json_file["data"]):
            num_insts += 1
            inst_id = inst["id"]
            hadm_id = inst["debug_features"]["HADM_ID"]
            dis_dt = inst["debug_features"]["DISCHTIME"].split(" ")[0].replace("-","")
            inst_images_meta = []
            pt_id = inst_id.split("-")[0]
            pt_shortid = pt_id[:2]
            pt_path = join(args[1], "files", "p"+pt_shortid, "p"+pt_id)
            if not exists(pt_path):
                #print("No images for patient %s" % (str(pt_id)))
                missing_images += 1
                continue

            for study_dir in os.scandir(pt_path):
                # study dirs are like s12345 where 12345 is the study number
                if study_dir.is_file() or study_dir.name.startswith('.'):
                    continue
                study_num = int(study_dir.name[1:])
                images = glob.glob(join(study_dir.path, "*.jpg"))
                for image_path in images:
                    # Image name is dicom_id.jpg
                    image_name = basename(image_path)
                    dicom_id = image_name[:-4]
                    inst_image_meta = meta_df[meta_df['dicom_id']==dicom_id].to_dict('record')[0]
                    study_dt = str(inst_image_meta["StudyDate"])
                    if study_dt <= dis_dt:      
                        inst_image_meta['path'] = "/".join(image_path.split("/")[-4:])
                        inst_images_meta.append(inst_image_meta)
        
            inst['images'] = inst_images_meta
    
    print("Processed %d instances and %d did not have corresponding images" % (num_insts, missing_images) )

    with open(args[2], 'wt') as fp:
        json.dump(json_file, fp, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])

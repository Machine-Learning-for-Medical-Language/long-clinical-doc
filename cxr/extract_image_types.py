import sys
import json


def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <input file (json)>\n")
        sys.exit(-1)

    with open(args[0], 'rt') as fp:
        data = json.load(fp)
        dataset_types = dict()
        inst_counts = 0
        for inst in data['data']:
            if 'images' not in inst or len(inst['images']) == 0:
                continue
            inst_counts += 1
            inst_types = set()
            adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
            sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
            for image in sorted_images:
                if str(image["StudyDate"]) >= adm_dt:
                    img_type = image["PerformedProcedureStepDescription"]
                    inst_types.add(img_type)

            for img_type in inst_types:
                if not img_type in dataset_types:
                    dataset_types[img_type] = 0
                
                dataset_types[img_type] += 1

    for key, val in dataset_types.items():
        print("%s => %d" % (key, val))

    print("Total number of instances with images: %d" % (inst_counts))

if __name__ == "__main__":
    main(sys.argv[1:])
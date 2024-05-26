import os, sys
from PIL import Image
from os.path import join
from tqdm import tqdm

MIN_RES=256

def main(args):
    if len(args) < 1:
        sys.stderr.write("Required argument(s): <mimic-cxr-jpg directory>\n")
        sys.exit(-1)

    with tqdm(total=371920) as pbar:
        for root, dirs, files in os.walk(args[0]):
            for file in files:
                if file.endswith(".jpg") and not file.endswith("_resized.jpg"):
                    ## Adapted from pytorch-cxr
                    ff = join(root, file)
                    img = Image.open(ff)
                    w, h = img.size
                    rs = (MIN_RES, int(h/w*MIN_RES)) if w < h else (int(w/h*MIN_RES), MIN_RES)
                    resized = img.resize(rs, Image.LANCZOS)
                    out_fn = file[:-4] + "_%d_resized.jpg" % (MIN_RES)
                    out_path = join(root, out_fn)
                    resized.save(out_path, "JPEG")
                    pbar.update(1)

if __name__ == "__main__":
    main(sys.argv[1:])
    
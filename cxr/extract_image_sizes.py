import os, sys
from PIL import Image
from os.path import join

def main(args):
    if len(args) < 1:
        sys.stderr.write("Reequired argument(s): <input directory>\n")
        sys.exit(-1)
    
    sizes = dict()
    for root, dirs, files in os.walk(args[0]):
        for file in files:
            if file.endswith(".jpg"):
                img = Image.open(join(root, file))
                size_str = ",".join([str(x) for x in img.size])
                if not size_str in sizes:
                    sizes[size_str] = 0
                    print("New size string found: %s" % (size_str))

                sizes[size_str] += 1

    for size,count in sizes.items():
        print("%s => %d" % (size, count))


if __name__ == "__main__":
    main(sys.argv[1:])
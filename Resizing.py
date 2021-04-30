#
# Function to resize NxN square images in folder.
# Creates a new folder for the images in the path
#
# Example:
# resizeImg(128, imgdir = "data/Train/TrainImages/", overwrite = False)
# Resizes images to 128x128 to folder data/Train/TrainImages/128/
#
# Overwrite
# if overwrite false, it will not resize if resize folder already exists
# if folder exists but not all images are resized, set to True

from PIL import Image
from os import mkdir, path, listdir

def resizeImg(size = 128, imgdir = "data/Train/TrainImages/", overwrite = False):
    dir = str(imgdir) + str(size) + "/"
    if not path.isdir(dir):
        print("resizing images to folder: "+ str(imgdir) + str(size) + "/")
        mkdir(dir)
        resizeFunc(128,str(imgdir))
    elif overwrite:
        print("overwriting existing images in: "+str(imgdir) + str(size) + "/")
        resizeFunc(128,str(imgdir))

def resizeFunc(size = 128, imgdir = ""):
    for x in range(1, len(listdir(imgdir))):
        i = Image.open(imgdir + "/Image" + str(x) + ".jpg")
        i = i.resize((128, 128), Image.BILINEAR)
        i.save(imgdir + str(size) + "/Image" + str(x) + ".jpg")



import os

import numpy

instances = []
from PIL import Image
import matplotlib.pyplot as plt
print("hello")
#def resizeImg(size):
from os import mkdir, path, getcwd

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
    for x in range(1, len(os.listdir(imgdir))):
        i = Image.open(imgdir + "/Image" + str(x) + ".jpg")
        i = i.resize((128, 128), Image.BILINEAR)
        i.save(imgdir + str(size) + "/Image" + str(x) + ".jpg")

resizeImg(128, True)
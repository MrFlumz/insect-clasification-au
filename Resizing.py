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
    dir = imgdir + str(size) + "/"
    if not path.isdir(dir):
        print("resizing images to folder: " + dir)
        mkdir(dir)
        resizeFunc(size,str(imgdir))
    elif overwrite:
        print("overwriting existing images in: "+ dir)
        resizeFunc(size,str(imgdir))

def resizeFunc(size = 128, imgdir = ""):
    nr_of_images = str(listdir(imgdir)).count("Image")
    print(nr_of_images)
    for x in range(1, nr_of_images+1):
        i = Image.open(imgdir + "Image" + str(x) + ".jpg")
        i = i.resize((size, size), Image.BILINEAR)
        printProgressBar(x,nr_of_images,length=30)
        i.save(imgdir + str(size) + "/Image" + str(x) + ".jpg")

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
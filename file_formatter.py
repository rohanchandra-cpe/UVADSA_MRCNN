import pandas as pd
import numpy as np
import shutil, errno
import os
from os.path import exists
import random

def rename_files(src, destination, json):
    """
    To properly name the Cogito files
    """
    if(not json): # FOR IMAGES,
        for dirpath, dirnames, files in os.walk(src): # ./Knot_Tying_Images_Unprocessed/Knot_Tying/G15\Knot_Tying_G005_373_567
            new_dirpath = dirpath.replace('\\', '/').split('/') # ['.', 'Knot_Tying_Images_Unprocessed', 'Knot_Tying', 'G15', 'KnotTying_G005_373_567']
            if(len(files) > 0):
                for file_name in files:
                    new_name = new_dirpath[2] + '_' + new_dirpath[3] + '_' + new_dirpath[4][10:] + '_' + file_name
                    os.rename(dirpath.replace('\\', '/') + "/" + file_name, destination + str(new_name))

    else: # FOR JSON
        for dirpath, dirnames, files in os.walk(src):
            new_dirpath = dirpath.replace('\\', '/').split('/')
            if(new_dirpath[3] == '' or new_dirpath[3] == 'classes'):
                continue
            if(len(files) > 0):
                for file_name in files:
                    teststr = ""
                    for i in range(3,6):
                        teststr += str(new_dirpath[3].split('_')[i] + "_") 
                    new_name = "Knot_Tying_" + str(new_dirpath[3].split('_')[0]) + "_" + teststr[:len(teststr) - 1] + "_" + str(file_name[0:10]) + ".json"
                    os.rename(dirpath.replace('\\', '/') + "/" + file_name, destination + str(new_name))

def train_test_split(train, test, validation, src):
    """
    A wrapper function for test_train_split()

    Figure out how to make and clear the folders every time this runs so that the values are properly updated
    """
    claimed_dict = {}
    file_size = len([file for file in os.listdir(src) if os.path.isfile(os.path.join(src, file))])
    train_size = (int)(train * file_size)
    test_size = (int)(test * file_size)
    validation_size = file_size - train_size - test_size

    if(not (exists("./train/images/"))):
        os.makedirs("./train/images/")
    if(not (exists("./train/json/"))):
        os.makedirs("./train/json/")
        
    if(not (exists("./test/images/"))):
        os.makedirs("./test/images/")
    if(not (exists("./test/json/"))):
        os.makedirs("./test/json/")

    if(not (exists("./validation/images/"))):
        os.makedirs("./validation/images/")
    if(not (exists("./validation/json/"))):
        os.makedirs("./validation/json/")

    for i in range(0, train_size):
        file_to_add = random.choice(os.listdir(src))
        try:
            while(claimed_dict[file_to_add] == 1): # To ensure no duplicates
                file_to_add = random.choice(os.listdir(src))
        except:
            claimed_dict[file_to_add] = 1
        shutil.copy(src + "/" + file_to_add, "./train/images/")
        shutil.copy("./json/" + file_to_add[:len(file_to_add) - 4] + ".json", "./train/json/")

    for i in range(0, test_size):
        file_to_add = random.choice(os.listdir(src))
        try:
            while(claimed_dict[file_to_add] == 1): # To ensure no duplicates
                file_to_add = random.choice(os.listdir(src))
        except:
            claimed_dict[file_to_add] = 1
        shutil.copy(src + "/" + file_to_add, "./test/images/")
        shutil.copy("./json/" + file_to_add[:len(file_to_add) - 4] + ".json", "./test/json/")

    for i in range(0, validation_size):
        file_to_add = random.choice(os.listdir(src))
        try:
            while(claimed_dict[file_to_add] == 1): # To ensure no duplicates
                file_to_add = random.choice(os.listdir(src))
        except:
            claimed_dict[file_to_add] = 1
        shutil.copy(src + "/" + file_to_add, "./validation/images/")
        shutil.copy("./json/" + file_to_add[:len(file_to_add) - 4] + ".json", "./validation/json/")

def copyanything(src, dst):
    print(exists(dst))
    try:
        if(exists(dst)):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

# rename_files("./Knot_Tying_Images_Unprocessed/Knot_Tying/", False)
# rename_files("./Knot_Tying_JSON_Unprocessed/Knot_Tying_Unzipped/", "", False)
# train_test_split(0.4, 0.4, 0.2)
# copyanything("./test_folder", "./new_folder")

def main():
    # rename_files("./Knot_Tying_Images_Unprocessed/Knot_Tying/", "./images/", False)
    # rename_files("./Knot_Tying_JSON_Unprocessed/Knot_Tying_Unzipped/", "./json/", True)
    train_test_split(0.4, 0.4, 0.2, "./images/")
if __name__ == "__main__":
    main()
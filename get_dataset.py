import os
import xml.etree.ElementTree as ET

import wget
import math
import tarfile
from tqdm import tqdm

import shutil
import csv
import glob



def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total) 
    return progress

def download(url, out_path='C:\\data\\pascalvoc'):
    wget.download(url, out=out_path, bar=bar_custom)
    print("\n")

def extract(url, out_path):
    # open your tar.gz file
    with tarfile.open(url) as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), 
                           total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member, path=out_path)
        tar.close()


# "https://pjreddie.com/media/files/voc_label.py"
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(path, year, image_id, classes):
    in_file = open(path + f'VOCdevkit/VOC{year}/Annotations/{image_id}.xml')
    out_file = open(path + f'VOCdevkit/VOC{year}/labels/{image_id}.txt', 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def voc_label():
    sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
               "car", "cat", "chair", "cow", "diningtable", "dog", 
               "horse", "motorbike", "person", "pottedplant", "sheep", 
               "sofa", "train", "tvmonitor"]   
    path_voc = 'C:/data/pascalvoc/'

    for year, image_set in sets:
        os.makedirs(path_voc + f'VOCdevkit/VOC{year}/labels/', exist_ok=True)
        image_ids = open(path_voc + f'VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt').read().strip().split()
        list_file = open(path_voc + f'{year}_{image_set}.txt', 'w')
        for image_id in image_ids:
            list_file.write(path_voc + f'VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg\n')
            convert_annotation(path_voc, year, image_id, classes)
        list_file.close()    

# https://github.com/aladdinpersson/...
def generate_csv(path):
    read_train = open(path + "train.txt", "r").readlines()

    with open(path + "train.csv", mode="w", newline="") as train_file:
        for line in read_train:
            image_file = line.split("/")[-1].replace("\n", "")
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)

    read_train = open(path + "test.txt", "r").readlines()

    with open(path + "test.csv", mode="w", newline="") as train_file:
        for line in read_train:
            image_file = line.split("/")[-1].replace("\n", "")
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)



def main():
    path_voc = 'C:/data/pascalvoc/'
    
    # VOC2007 DATASET
    url_voc2007_trainval = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    url_voc2007_test = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    
    # VOC2012 DATASET
    url_voc2012_trainval = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    
    # Download VOC DATASETS
    download(url_voc2007_trainval)
    download(url_voc2007_test)
    download(url_voc2012_trainval)

    # Extract tar files
    extract(path_voc + "VOCtrainval_06-Nov-2007.tar", path_voc)
    extract(path_voc + "VOCtest_06-Nov-2007.tar", path_voc)
    extract(path_voc + "VOCtrainval_11-May-2012.tar", path_voc)

    # Clean up data from xml files
    # download("https://pjreddie.com/media/files/voc_label.py")
    voc_label()

    ## Organize files
    # Define the list of files
    file_list = ['2007_train.txt', '2007_val.txt', '2012_train.txt', '2012_val.txt']
    file_test = '2007_test.txt' 

    # Create and open a new file - train.txt
    with open(path_voc + 'train.txt', 'w') as outfile:
        for fname in file_list:
            with open(path_voc + fname) as infile:
                outfile.write(infile.read())
            infile.close()
    outfile.close()

    # Create and open a new file - test.txt
    with open(path_voc + 'test.txt', 'w') as outfile:
        with open(path_voc + file_test) as infile:
            outfile.write(infile.read())
        infile.close()
    outfile.close()

    # Move txt files we won't be using to clean up a little bit
    os.makedirs(path_voc + "old_txt_files")

    for fname in file_list:
        shutil.move(os.path.join(path_voc, fname), os.path.join(path_voc, 'old_txt_files'))
    shutil.move(os.path.join(path_voc, file_test), os.path.join(path_voc, 'old_txt_files'))

    # Create csv files (columns : jpg file name, txt file name)
    generate_csv(path_voc)

    # Create a data folder used for training, validation, testing
    os.makedirs(path_voc + "data")
    os.makedirs(path_voc + "data/images")
    os.makedirs(path_voc + "data/labels")

    # Move files *.jpg
    jpg_sources = ['VOCdevkit/VOC2007/JPEGImages/*.jpg', 'VOCdevkit/VOC2012/JPEGImages/*.jpg']
    for source_pattern in jpg_sources:
        for file in glob.glob(path_voc + source_pattern):
            shutil.move(file, path_voc + 'data/images/' + file.split('\\')[-1])

    # Move files *.txt
    txt_sources = ['VOCdevkit/VOC2007/labels/*.txt', 'VOCdevkit/VOC2012/labels/*.txt']
    for source_pattern in txt_sources:
        for file in glob.glob(path_voc + source_pattern):
            shutil.move(file, path_voc + 'data/labels/' + file.split('\\')[-1])

    # We don't need VOCdevkit folder anymore, can remove
    shutil.rmtree(path_voc + "VOCdevkit", ignore_errors=True)
    shutil.move(os.path.join(path_voc, 'train.txt'), os.path.join(path_voc, 'old_txt_files'))
    shutil.move(os.path.join(path_voc, 'test.txt'), os.path.join(path_voc, 'old_txt_files'))



if __name__ == "__main__":
    main()
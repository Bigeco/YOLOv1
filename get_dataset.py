import os
import xml.etree.ElementTree as ET

import wget
import math
import tarfile
from tqdm import tqdm

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

def main():
    path_voc = 'C:\\data\\pascalvoc'
    
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
    extract(path_voc + "\\VOCtrainval_06-Nov-2007.tar", path_voc)
    extract(path_voc + "\\VOCtest_06-Nov-2007.tar", path_voc)
    extract(path_voc + "\\VOCtrainval_11-May-2012.tar", path_voc)

    # Clean up data from xml files
    # download("https://pjreddie.com/media/files/voc_label.py")
    voc_label()

if __name__ == "__main__":
    main()
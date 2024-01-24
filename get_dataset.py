import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import VOCDetection

def download_pascal_voc(root='C:/data/pascalvoc'):
    # Download and Load the Pascal VOC 2007 Dataset
    voc_2007 = VOCDetection(root=root + "/2007", 
                            year='2007', 
                            image_set='train', 
                            download=True)

    # Download and Load the Pascal VOC 2012 Dataset
    voc_2012 = VOCDetection(root=root + "/2012", 
                            year='2012', 
                            image_set='train', 
                            download=True)  
     
    return voc_2012, voc_2012 

def main():
    voc_2007, voc_2012 = download_pascal_voc()

if __name__ == "__main__":
    main()
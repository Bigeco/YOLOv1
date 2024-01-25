import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import VOCDetection

def download_pascal_voc(root='C:\\data\\pascalvoc'):
    # Download and Extract the Pascal VOC 2007 Dataset
    VOCDetection(root=root + "\\2007", 
                            year='2007', 
                            image_set='train', 
                            download=True)

    # Download and Extract the Pascal VOC 2012 Dataset
    VOCDetection(root=root + "\\2012", 
                            year='2012', 
                            image_set='train', 
                            download=True)  

def main():
    download_pascal_voc()

if __name__ == "__main__":
    main()
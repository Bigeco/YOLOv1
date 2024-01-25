import wget
import math
import tarfile
from tqdm import tqdm


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

if __name__ == "__main__":
    main()
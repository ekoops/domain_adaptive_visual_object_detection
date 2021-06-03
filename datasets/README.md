# Overview
We provide three datasets (Pascal VOC 2007, Pascal VOC 2012, Clipart1k).

Please refer to the paper for further details about the datasets.

# Details for the datasets
The Pascal VOC 2007 dataset contains both trainval split and the test split.
The Pascal VOC 2012 dataset contains the trainval split.
The Clipart1k datasets contains both the train and the test split.

# Setup

## Download original datasets
Before executing the following command, `curl`, `rsync`, `tar` and `unzip` tools have to be available.
Execute the following command to download all the necessary datasets.
```
bash prepare.sh
```

The above command will create under this folder the following folder structure:

```
VOC
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
CLIPART
|__ clipart
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
```
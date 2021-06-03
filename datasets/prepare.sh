#!/usr/bin/env bash

VOC2007_trainval_archive=VOCtrainval_06-Nov-2007.tar
VOC2007_trainval_url=http://host.robots.ox.ac.uk/pascal/VOC/voc2007/${VOC2007_trainval_archive}

VOC2007_test_archive=VOCtest_06-Nov-2007.tar
VOC2007_test_url=http://host.robots.ox.ac.uk/pascal/VOC/voc2007/${VOC2007_test_archive}

VOC2012_trainval_archive=VOCtrainval_11-May-2012.tar
VOC2012_trainval_url=http://host.robots.ox.ac.uk/pascal/VOC/voc2012/${VOC2012_trainval_archive}

Clipart1k_archive=clipart.zip
Clipart1k_url=http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/${Clipart1k_archive}

mkdir VOC

# Download VOC2007 trainval split
echo "Downloading VOC2007 trainval split..."
curl -O ${VOC2007_trainval_url}
echo "Extracting VOC2007 trainval split..."
tar -xf ${VOC2007_trainval_archive} -C ./VOC --strip-components=1
rm -r ${VOC2007_trainval_archive}

# Download VOC2007 test split and merge with the trainval split
echo "Downloading VOC2007 test split..."
curl -O ${VOC2007_test_url}
echo "Extracting VOC2007 test split..."
tar -xf ${VOC2007_test_archive} --strip-components=1
echo "Merging VOC2007 test split with the trainval split..."
rsync -a ./VOC2007/ ./VOC/VOC2007/
rm -r ${VOC2007_test_archive}
rm -r ./VOC2007

# Download VOC2012 trainval split
echo "Downloading VOC2012 trainval split..."
curl -O ${VOC2012_trainval_url}
echo "Extracting VOC2012 trainval split..."
tar -xf ${VOC2012_trainval_archive} -C ./VOC --strip-components=1
rm -r ${VOC2012_trainval_archive}

mkdir CLIPART

# Download Clipart1k
echo "Downloading Clipart1k..."
curl -O ${Clipart1k_url}
echo "Extracting Clipart1k..."
unzip ${Clipart1k_archive} -d ./CLIPART
rm -r ${Clipart1k_archive}
echo "Done."
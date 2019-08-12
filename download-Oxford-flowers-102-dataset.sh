#!/usr/bin/env bash
# Script to download and extract the dataset: Flower species 102 (UK)
# See: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

# cd to your folder where you want to save the data.
cd "$1"
mkdir Oxford-flowers-102
cd Oxford-flowers-102

# Download the images.
echo "Downloading dataset images (329MB) ..."
wget -O 102flowers.tgz http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# Download masks (image segmentations)
echo "Downloading segmentation (194MB) ..."
wget -O 102segmentations.tgz http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz


# Download the the image labels
echo "Downloading the image labels ..."
wget -O imagelabels.mat http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# Download the splits
wget -O setid.mat http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

# Download the readme
wget -O README.txt  http://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt


echo "Finished downloading Oxford-flowers-102 dataset."

echo "Extracting files ..."

tar -zxvf 102flowers.tgz
tar -zxvf 102segmentations.tgz
ls


echo "Finished extracting Oxford-flowers-102 dataset."
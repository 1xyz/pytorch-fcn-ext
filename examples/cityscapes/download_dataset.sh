#!/bin/bash

DIR=~/data/datasets/CityScapes/CityScapes/

mkdir -p $DIR
cd $DIR

# Labels
gsutil cp gs://bhram-test-datasets/cityscapes/gtFine_trainvaltest.zip ./gtFine_trainvaltest.zip
unzip -nq gtFine_trainvaltest.zip

# Images
gsutil cp gs://bhram-test-datasets/cityscapes/leftImg8bit_trainvaltest.zip ./leftImg8bit_trainvaltest.zip
unzip -nq leftImg8bit_trainvaltest.zip

#!/bin/bash

PYTHON=/usr/bin/python3

echo "Downloading data"
if [ ! -d "data/" ]; then
  mkdir data
fi
wget -P data/ https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
# wget -P data/ https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
# wget -P data/ https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz

echo "Unzip file"
unzip -o -d data/ data/dogImages.zip
rm data/dogImages.zip
# unzip -o -d data/ data/lfw.zip
# rm data/lfw.zip


echo "Loading data"
echo "----------------------------"
"$PYTHON" 01_01_load_files.py

# echo "Data preprocess"
# echo "----------------------------"
# "$PYTHON" 04_dog_detector_preprocess.py

echo "Training data"
echo "----------------------------"
"$PYTHON" 04_dog_detector_train.py

# echo "Test data" # 使用测试集评价模型
# echo "----------------------------"
# "$PYTHON" 04_dog_detector_test.py

echo "Run app"
echo "----------------------------"
"$PYTHON" 05_dog_app.py

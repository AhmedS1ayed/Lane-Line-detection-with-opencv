#!/bin/bash

#install python and pip
sudo apt-get install python3-pip
sudo apt-get install python3-venv

#create virtual enviroment
python3 -m venv env

#activate virtual enviroment
source env/bin/activate

#install dependencies
sudo apt-get install libjpeg-dev zlib1g-dev
python3 -m pip install pillow
python3 -m pip install matplotlib
 python3 -m pip install scikit-build
pip install --upgrade pip setuptools wheel
python3 -m pip install opencv-python
python3 -m pip install imageio
python3 -m pip install moviepy
python3 -m pip install IPython
python3 -m pip install getpy
python3 Lane_line_detection.py $1 $2 $3
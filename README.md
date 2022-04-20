# Lane-Line-detection-with-opencv
this project solve the lane line detection problem using opencv

# How to use 
use the bash script as following :
$ bash bashscript.sh input_file.mp4 output_file.mp4 --d

Note : 1- you can use -h for help .
       2- --d is for debugging .
       
# Dependencies 

sudo apt-get install libjpeg-dev zlib1g-dev
python3 -m pip install pillow
python3 -m pip install matplotlib
python3 -m pip install opencv-python

python3 Lane_line_detection.py $1 $2 $3

Note : No need to install them just run the bash file and it will generate the virtual enviroment with the dependencies

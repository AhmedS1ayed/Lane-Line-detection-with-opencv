#!/bin/bash
if [ "$1" == "k3" ]; then
    project_path="/Lane-Line-detection-with-opencv-main"
fi
code_path="code"
full_path="$project_path$code_path"
# Go to directory of project
cd $full_path
# Start environment & notebook if available
pipenv shell
pip install jupyter notebook
pipenv run jupyter notebook
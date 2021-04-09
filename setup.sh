#!/bin/bash
echo "Setting up LaneNet environment..."
python3 -m virtualenv --python=python3 --clear venv-lanenet
source venv-lanenet/bin/activate
python3 -m pip install --upgrade pip

echo "Making a new folder for the neural network weights..."
mkdir -p model_weights
cd model_weights
echo "Downloading neural network model..."
wget "https://www.dropbox.com/sh/0b6r0ljqi76kyg9/AADedYWO3bnx4PhK1BmbJkJKa?dl=1" --output-document bisenetv2_lanenet_model_weights.zip
unzip bisenetv2_lanenet_model_weights.zip -d bisenetv2_lanenet_model_weights
cd ..

PYTHON3_VERSION_INSTALLED=$(python3 --version | awk '{print $2}')
HAS_PYTHON_36=$(echo ${PYTHON3_VERSION_INSTALLED} | grep "^3.6" | wc -l)

if [ ${HAS_PYTHON_36} -gt 0 ]; then
    echo "Python ${PYTHON3_VERSION_INSTALLED} found. Installing Python requirements..."
    python3 -m pip install -r requirements.txt
else
    echo "Python 3.6 not found! Can't proceed with the requirements installation."
    echo "Please proceed with python dependency installation one by one,"
    echo "since package version compatibility can't be guaranteed between Python versions."
fi

echo "Adding the repository root to the PYTHONPATH..."
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "Environment setup done."

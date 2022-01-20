#!/usr/bin/env bash

source /opt/anaconda3/etc/profile.d/conda.sh
conda update -n base -c defaults conda
conda create --name openvino_env
conda activate openvino_env

cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
./install_prerequisites.sh
pip install --upgrade pip

cd ~/Desktop
git clone https://github.com/openvinotoolkit/openvino_notebooks.git
cd ~/Desktop/openvino_notebooks
pip install -r requirements.txt


conda install openvino-ie4py -c intel
conda install -c anaconda ipykernel
pip install ipykernel
pip install matplotlib
pip install opencv-python
cd /opt/anaconda3/lib/python3.8/site-packages
cp -r /opt/intel/openvino_2021.4.689/python/python3.8/openvino .
python -m ipykernel install --user --name openvino_env

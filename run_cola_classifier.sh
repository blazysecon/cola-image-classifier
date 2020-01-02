#! /bin/bash
set -e

cd ~/data_disk/pers_proj/cola-image-classifier
~/anaconda3/envs/fastai/bin/python app/server.py serve

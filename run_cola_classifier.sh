#! /bin/bash
set -e

cd /home/giedrbl47/data_disk/pers_proj/cola-image-classifier
/home/giedrbl47/anaconda3/envs/fastai/bin/python app/server.py serve

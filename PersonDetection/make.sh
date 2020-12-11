#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/home/caiting/cuda10_1

python build.py build_ext --inplace

cd ..

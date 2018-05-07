#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -e err.txt
#$ -o log.txt
#$ -ac d=nvcr-cuda-9.0-cudnn7
#$ -jc gpu-container_g1_dev

export PATH=~/anaconda3/bin:$PATH

source activate test

python -m allennlp.run_elmo --config_fn elmo_2x4096_512_2048cnn_2xhighway_options.json --weight_fn elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 --in_fn sentences.txt --out_fn sentences.elmo.hdf5

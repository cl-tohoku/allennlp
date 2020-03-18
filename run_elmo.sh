#!/bin/bash

source activate allennlp
python -m allennlp.run_elmo --cuda_device 0 --option_fn path/to/option.json --weight_fn path/to/weights.hdf5 --in_fn path/to/input_data --out_fn elmo.hdf5

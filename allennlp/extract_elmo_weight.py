#!/usr/bin/env python
import h5py


def save_h5(fin, key="#", value=[]):
    if key in fin and type(fin[key]) == h5py.Dataset:
        return value + [(key, fin[key][...])]

    if key == "#":
        f_list = list(fin)
        key = ""
    else:
        f_list = list(fin[key].keys())

    for k in f_list:
        value = save_h5(fin, key + "/" + k, value)
    return value


if __name__ == "__main__":
    from allennlp.common.file_utils import cached_path

    DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"  # pylint: disable=line-too-long
    fin = h5py.File(cached_path(DEFAULT_WEIGHT_FILE), 'r')
    fout = h5py.File("elmo.weight.hdf5", 'w')
    for (k, v) in save_h5(fin):
        fout.create_dataset(name=k, data=list(v))
    fin.flush()
    fin.close()
    fout.flush()
    fout.close()

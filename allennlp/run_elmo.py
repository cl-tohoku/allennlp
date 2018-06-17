#!/usr/bin/env python
import logging
import os
import sys
import locale

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

if __name__ == "__main__":
    import argparse
    import h5py
    from allennlp.commands.elmo import ElmoEmbedder

    parser = argparse.ArgumentParser(description='ELMo Embedder.')

    parser.add_argument('--option_fn', default='model_files/elmo.config.json', help='path to option file')
    parser.add_argument('--weight_fn', default='model_files/elmo.weight.hdf5', help='path to weight file')
    parser.add_argument('--cuda_device', type=int, default=-1, help='cuda_device')
    parser.add_argument('--in_fn', default=None, help='path to input file')
    parser.add_argument('--out_fn', default='elmo_embs.hdf5', help='path to output file')
    parser.add_argument('--data_size', type=int, default=100000000, help='data size')

    argv = parser.parse_args()

    if argv.cuda_device > -1:
        locale.setlocale(locale.LC_CTYPE, 'C.UTF-8')

    sys.stdout.write("Build ELMo Embedder\n")
    sys.stdout.write("Use CUDA: %d\n" % argv.cuda_device)
    sys.stdout.flush()
    ee = ElmoEmbedder(options_file=argv.option_fn,
                      weight_file=argv.weight_fn,
                      cuda_device=argv.cuda_device)

    sys.stdout.write("Prediction START\n")
    sys.stdout.flush()
    outfh = h5py.File(argv.out_fn, 'w')
    with open(argv.in_fn, 'r') as f:
        sent_index = 0
        sys.stdout.write("Sent: ")
        sys.stdout.flush()

        for line in f:
            line = line.rstrip()
            embeddings = ee.embed_sentence(line.split())
            outfh.create_dataset(name=str(sent_index),
                                 data=embeddings)

            sent_index += 1
            if sent_index % 100 == 0:
                sys.stdout.write("%d " % sent_index)
                sys.stdout.flush()

            if sent_index >= argv.data_size:
                break

    sys.stdout.write("\nPrediction FINISHED\n")
    outfh.flush()
    outfh.close()

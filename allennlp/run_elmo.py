#!/usr/bin/env python
import logging
import os
import sys
import locale

locale.setlocale(locale.LC_CTYPE, 'C.UTF-8')

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

    parser.add_argument('--config_fn', default='model_files/elmo.config.json', help='path to config file')
    parser.add_argument('--weight_fn', default='model_files/elmo.weight.hdf5', help='path to weight file')
    parser.add_argument('--in_fn', default=None, help='path to input file')
    parser.add_argument('--out_fn', default='elmo_embs.hdf5', help='path to output file')

    argv = parser.parse_args()

    sys.stdout.write("Build ELMo Embedder\n")
    sys.stdout.flush()
    ee = ElmoEmbedder(argv.config_fn, argv.weight_fn)

    sys.stdout.write("Prediction START\n")
    sys.stdout.flush()
    outfh = h5py.File(argv.out_fn, 'w')
    with open(argv.in_fn, 'r') as f:
        sent_id = 0
        for line in f:
            embeddings = ee.embed_sentence(line.rstrip().split())
            outfh.create_dataset(str(sent_id), data=embeddings)
            sent_id += 1
            if sent_id % 100 == 0:
                sys.stdout.write("At Sent %d\n" % sent_id)
                sys.stdout.flush()
    outfh.flush()
    outfh.close()

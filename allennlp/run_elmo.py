#!/usr/bin/env python
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

if __name__ == "__main__":
    from allennlp.commands.elmo import ElmoEmbedder

    ee = ElmoEmbedder(sys.argv[1], sys.argv[2])
    embeddings = ee.embed_sentence("Bitcoin alone has a sixty percent share of global search .".split())
    print(embeddings.shape)

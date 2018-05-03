# Memo

## 04/27
- The gamma value in Eq.1 is in ScalarMix (allennlp/modules/scalar_mix.py)
- When using Pytorch 0.4.0, a segmentation fault error appears in block_orthogonal in initializers.py

## 05/03
- The ELMo in pytorch cannot return probability distributions for the output vocabulary.
- We cannot check the perplexity on a dataset using the ELMo.

- Evaluate an SRL model
```
python -m allennlp.run evaluate --evaluation-data-file data/conll12_test_files srl-model-2018.02.27.tar.gz
```

- Train simple tagger with ELMo
```
python -m allennlp.run train training_config/simple_tagger_elmo.json --serialization-dir /tmp/tutorials/getting_started
```

- Train simpler tagger with ELMo
```
python -m allennlp.run train training_config/simpler_tagger_elmo.json --serialization-dir /tmp/tutorials/getting_started
```

- Predict ELMo embeddings
```
python -m allennlp.run elmo data/sentences.txt elmo_layers.hdf5 --all
```

- See the hdf5 file of the ELMo embeddings
```
python allennlp/view_hdf.py elmo_layers.hdf5
```

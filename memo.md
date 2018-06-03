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

- Predict ELMo embeddings
```
python -m allennlp.run elmo sent.txt elmo_layers.hdf5 --all
```

-- Predict and Save ELMo embeddings:
```
python -m allennlp.run_elmo --in_fn sent.txt --out_fn sent.hdf5 --option_fn options.json --weight_fn weights.hdf5
```

- See the hdf5 file of the ELMo embeddings
```
python allennlp/view_hdf.py elmo_layers.hdf5
```

# Memo

## 04/27
- The gamma value in Eq.1 is in ScalarMix (allennlp/modules/scalar_mix.py)
- When using Pytorch 0.4.0, a segmentation fault error appears in block_orthogonal in initializers.py

## 05/03
- The ELMo in pytorch cannot return probability distributions for the output vocabulary.
- We cannot check the perplexity on a dataset using the ELMo.

- Evaluation for SRL
```
python -m allennlp.run evaluate --evaluation-data-file data/conll12_test_files srl-model-2018.02.27.tar.gz
```

- Training for Simple Tagger with ELMo
```
python -m allennlp.run train training_config/simple_tagger_elmo.json --serialization-dir /tmp/tutorials/getting_started
```

- Training for Simpler Tagger with ELMo
```
python -m allennlp.run train training_config/simpler_tagger_elmo.json --serialization-dir /tmp/tutorials/getting_started
```
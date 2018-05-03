# Memo

## 04/27
- The gamma value in Eq.1 is in ScalarMix (allennlp/modules/scalar_mix.py)
- When using Pytorch 0.4.0, a segmentation fault error appears in block_orthogonal in initializers.py

## 05/03
- The ELMo in pytorch cannot return probability distributions for the output vocabulary.
- We cannot check the perplexity on a dataset using the ELMo.
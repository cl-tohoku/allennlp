# Getting started at Raiden

## 1. Install Anaconda
	bash Anaconda3-4.3.0-Linux-x86_64.sh
	export PATH=~/anaconda3/bin:$PATH
	conda create -n allennlp python=3.6
	source activate allennlp

## 2. Clone allennlp
	git clone https://github.com/allenai/allennlp.git
	INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh

## 3. Install cuda and pytorch
	conda install pytorch torchvision cuda90 -c pytorch

## 4. Install a dataset
	wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
	unzip brown.zip

## 5. Rewrite the config file for running allennlp/models/simple_tagger.py
	emacs tutorials/getting_started/simple_tagger.json
- rewrite:
	"train_data_path": "brown/ca01",
	"validation_data_path": "brown/ca02",
- rewrite if you want to use GPU:
	"cuda_device": 0

## 6. Run the following command
	python -m allennlp.run train tutorials/getting_started/simple_tagger.json --serialization-dir /tmp/tutorials/getting_started

- Please refer to https://github.com/allenai/allennlp/blob/v0.4.0/tutorials/getting_started/training_and_evaluating.md


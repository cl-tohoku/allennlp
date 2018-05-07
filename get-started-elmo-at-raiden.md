# Getting started @ Raiden

## 1. Install Anaconda
    wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
	bash Anaconda3-5.1.0-Linux-x86_64.sh
	export PATH=~/anaconda3/bin:$PATH
	conda create -n allennlp python=3.6
	source activate allennlp

## 2. Install cuda and pytorch
	conda install pytorch torchvision cuda90 -c pytorch

## 3. Clone allennlp
	git clone https://github.com/cl-tohoku/allennlp.git
	INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh

## 4. Download config and params for ELMo
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
    wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

## 5. Prepare sentences
    echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
    echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt

## 6. Run a script for the Raiden batch job
    qsub ELMO.sh

## 7. See the resulting file
    python allennlp/view_hdf.py sentences.elmo.hdf5


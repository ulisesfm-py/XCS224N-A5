#!/bin/bash

if [ "$1" = "train" ]; then
  mkdir -p trained_models
	CUDA_VISIBLE_DEVICES=0 python run.py train --save-to=./trained_models/model_soln.bin --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=./vocabs/vocab_soln.json --cuda --batch-size=2

elif [ "$1" = "test" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_soln.txt
    CUDA_VISIBLE_DEVICES=0 python run.py decode trained_models/model_soln.bin ./en_es_data/test.es outputs/test_outputs_soln.txt --cuda

elif [ "$1" = "train_local" ]; then
  mkdir -p trained_models
  	python run.py train --save-to=./trained_models/model_soln.bin --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en \
        --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=./vocabs/vocab_soln.json --batch-size=2

elif [ "$1" = "test_local" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_soln.txt
    python run.py decode trained_models/model_soln.bin ./en_es_data/grader.es outputs/test_outputs_soln.txt

elif [ "$1" = "train_local_q1" ]; then
  mkdir -p trained_models
	python run.py train --save-to=./trained_models/model_local_q1_soln.bin --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=./vocabs/vocab_tiny_q1_soln.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder
elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q1.txt
    python run.py decode trained_models/model_local_q1_soln.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1_soln.txt \
        --no-char-decoder
elif [ "$1" = "train_local_q2" ]; then
  mkdir -p trained_models
	python run.py train --save-to=./trained_models/model_local_q2_soln.bin --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=./vocabs/vocab_tiny_q2_soln.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100
elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_outputs_local_q2.txt
    python run.py decode trained_models/model_local_q2_soln.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2_soln.txt 
elif [ "$1" = "vocab" ]; then
    python run.py vocab --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en --size=200 --freq-cutoff=1 vocabs/vocab_tiny_q1_soln.json
    python run.py vocab --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en                            vocabs/vocab_tiny_q2_soln.json
    python run.py vocab --train-src=./en_es_data/train.es      --train-tgt=./en_es_data/train.en                                 vocabs/vocab_soln.json
else
	echo "Invalid Option Selected"
fi

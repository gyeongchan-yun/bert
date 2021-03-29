#!/bin/bash

export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
export PATH=~/.openmpi_4.0.0/bin:$PATH
export LD_LIBRARY_PATH=~/.openmpi_4.0.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH+=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH+=/usr/local/cuda-10.0/extras/CUPTI/lib64/

total_num_gpus=4
cluster_topology=shark8:4

BERT_BASE_DIR=~/bert_gc/pre-trained/base/cased_L-12_H-768_A-12
BERT_LARGE_DIR=~/bert_gc/pre-trained/large/cased_L-24_H-1024_A-16
SQUAD_DIR=/ssd_dataset/dataset/squad

PRE_TRAINED_DIR=$BERT_LARGE_DIR
#PRE_TRAINED_DIR=$BERT_BASE_DIR

source ~/tf1.12_py3/bin/activate

mpirun -np $total_num_gpus \
    -H $cluster_topology \
    --prefix ~/.openmpi_4.0.0 \
    -bind-to none -map-by slot \
    -x PATH -x LD_LIBRARY_PATH \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ib0 \
    python run_squad_hvd.py \
    --vocab_file=$PRE_TRAINED_DIR/vocab.txt \
    --bert_config_file=$PRE_TRAINED_DIR/bert_config.json \
    --init_checkpoint=$PRE_TRAINED_DIR/bert_model.ckpt \
    --do_train=True \
    --train_file=$SQUAD_DIR/train-v2.0.json \
    --do_predict=True \
    --predict_file=$SQUAD_DIR/dev-v2.0.json \
    --train_batch_size=24 \
    --learning_rate=3e-5 \
    --num_train_epochs=2.0 \
    --max_seq_length=256 \
    --doc_stride=128 \
    --output_dir=./squad2.0/ \
    --version_2_with_negative=True \
    --do_lower_case=False

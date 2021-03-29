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
GLUE_DIR=/ssd_dataset/dataset/glue

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
    python run_classifier_hvd.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=4.0 \
    --output_dir=./mrpc_output/ \
    --do_lower_case=False

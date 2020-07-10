#!/bin/bash
echo 'Run evaluation...'
python eval.py \
    --cuda \
    --vocab_file vocab/Firehose_cased_32000.model \
    --dataset_path data/Firehose100M_dataset/ \
    --dataset Firehose100M \
    --split test \
    --same_length \
    --batch_size 64 \
    --cased \
    ${@:1}

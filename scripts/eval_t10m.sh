#!/bin/bash
echo 'Run evaluation...'
python eval.py \
    --cuda \
    --vocab_file vocab/Firehose_cased_32000.model \
    --dataset_path data/Firehose10M/dataset/ \
    --dataset Firehose10M \
    --split test \
    --same_length \
    --batch_size 128 \
    --cased \
    ${@:1}

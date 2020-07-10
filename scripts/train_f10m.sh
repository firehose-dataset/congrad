#!/bin/bash

# constant parameters
WORK_DIR='exp/MTLTransformer'
N_LAYER=12
D_MODEL=512
N_HEAD=8
D_HEAD=64
D_USER_EMBED=32
D_INNER=2048
DROPOUT=0.1
DROPATT=0.1
OPTIMIZER='adam'
SCHEDULER='constant'
LR=0.0001
SEGLEN=50
INIT_STD=0.0001
MTL_Width=0.5
MTL_Depth=1

DATASET='Firehose10M'
WARMUP=2000
LOG_INTERVAL=100
EVAL_INTERVAL=2000
EVAL_STEPS=1000
BASE_LR=0.0001


echo 'Run training...'
python train.py \
    --cuda \
    --work_dir $WORK_DIR \
    --model_class MTLMemTransformerLM \
    --vocab_file vocab/Firehose_cased_32000.model \
    --dataset_path data/$DATASET/dataset/ \
    --dataset $DATASET \
    --n_layer $N_LAYER \
    --d_user_embed $D_USER_EMBED \
    --d_model $D_MODEL \
    --n_head  $N_HEAD \
    --d_head  $D_HEAD \
    --d_inner $D_INNER \
    --dropout $DROPOUT \
    --dropatt $DROPATT \
    --optim   $OPTIMIZER \
    --lr      $LR\
    --warmup_step   $WARMUP \
    --init_std $INIT_STD \
    --max_step      -1 \
    --max_seqlen    $SEGLEN \
    --scheduler     $SCHEDULER \
    --log-interval  $LOG_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --max_eval_steps $EVAL_STEPS \
    --eta_min       1e-5 \
    --batch_size        $1 \
    --eval_batch_size   256 \
    --online_batch_size $2 \
    --replay_batch_size $3 \
    --online_per_user_rbsize $4 \
    --replay_per_user_rbsize $5 \
    --online_buffer_strategy greedy \
    --mtl-type layerwise \
    --mtl_width $MTL_Width \
    --mtl_depth $MTL_Depth \
    --max_k_steps $6 \
    --lr $BASE_LR \
    --cased \
    ${@:7}

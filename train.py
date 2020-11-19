# coding: utf-8
import argparse
import json
import time
import math
import os, sys
import itertools

import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from core.dataset.corpus import get_lm_corpus
from core.configs import get_basic_parser
from core.trainer import OnlineTrainer, batch_evaluate

#TODO: Dangerous line of code below, make sure remove it when you don"t know what you ignored
import warnings
warnings.filterwarnings("ignore")

def postprocess_args(args):
    args.tie_weight = not args.not_tied
    args.d_embed = args.d_model if args.d_embed < 0 else args.d_embed
    args.d_user_embed = args.d_embed if args.d_user_embed < 0 else args.d_user_embed
    assert args.batch_size % args.batch_chunk == 0

    if args.snapshot_dir is not None:
      with open(os.path.join(args.snapshot_dir, "configs.json")) as fd:
        max_step = args.max_step
        snapshot_dir = args.snapshot_dir
        args_json = json.load(fd)
        args = argparse.Namespace(**args_json)
        args.max_step = max_step
        args.snapshot_dir = snapshot_dir
    else:
        args.work_dir = "_".join([ _ for _ in [
                                args.work_dir,
                                args.dataset,
                                "cased" if args.cased else None,
                            ] if _ is not None ])
        args.work_dir = os.path.join( args.work_dir, "_".join([ _ for _ in [
                                      args.learner,
                                      "{}_online".format(args.online_buffer_strategy),
                                      "{}_replay".format(args.replay_buffer_strategy),
                                      "{:03g}databsz".format(args.batch_size),
                                      "{:03g}obsz".format(args.online_batch_size),
                                      "{:03g}rbsz".format(args.replay_batch_size),
                                      "{:02g}opusize".format(args.online_per_user_rbsize),
                                      "{:02g}rpusize".format(args.replay_per_user_rbsize),
                                      "{:02g}maxk".format(args.max_k_steps),
                                      "{}".format(args.mtl_type) if args.model_class.startswith("MTL") else None,
                                      "allowZeroStep" if args.allow_zero_step else None,
                                      "fromPretrained" if args.init_weights is not None else None,
                                      args.postfix,
                                ] if _ is not None ]),
                            )
        args.work_dir = os.path.join( args.work_dir, "_".join([ _ for _ in [
                                      args.model_class,
                                      "{}".format(args.mtl_type) if args.model_class.startswith("MTL") else None,
                                      "maxlen{:03d}".format(args.max_seqlen),
                                      "lr{:.4g}".format(args.lr),
                                      "time{}".format(time.strftime("%Y%m%d_%H%M%S"))
                                ] if _ is not None ]),
                            )

    return args

def _command_line_parser():
    parser = argparse.ArgumentParser(parents=[get_basic_parser()])
    parser.add_argument("--dataset_path", type=str, default="data/Firehose10M",
                        help="location of the data corpus")
    parser.add_argument("--dataset", type=str, default="Firehose10M",
                        help="dataset name")
    parser.add_argument("--cased", default=False, action="store_true",
                        help="use cased or uncased corpus")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="vocabulary")

    # replay buffer configs
    parser.add_argument("--online_batch_size", type=int, default=128,
                        help="online batch size")
    parser.add_argument("--online_per_user_rbsize", type=int, default=1,
                        help="per user online memory buffer size")
    parser.add_argument("--replay_batch_size", type=int, default=128,
                        help="replay batch size")
    parser.add_argument("--replay_per_user_rbsize", type=int, default=8,
                        help="per user replay memory buffer size")
    parser.add_argument("--online_buffer_strategy", type=str, default="greedy",
                        help="online cache strategy (default: greedy)",
                        choices=["greedy", "reservoir", "stratified", "stratified-reservoir"])
    parser.add_argument("--replay_buffer_strategy", type=str, default="greedy",
                        help="replay buffer strategy (default: greedy)",
                        choices=["greedy", "reservoir", "stratified", "stratified-reservoir"])

    parser.add_argument("--allow_zero_step", action="store_false",
                        help="whether allow the minimum number of gradient steps to be zero in ConGraD.")
    parser.add_argument("--max_k_steps", type=int, default=1,
                        help="the maximum number of gradient steps per online data chunk.")
    parser.add_argument("--learner", type=str, default="OnlineOnly",
                        help="type of online learning algorithms",
                        choices=["AGEM", "OnlineOnly", "ReplayOnly", "MixedReplay",
                                 "ConGraD_AGEM", "ConGraD_OnlineOnly", "ConGraD_ReplayOnly",
                                 "ConGraD_MixedReplay"])

    return parser

if __name__ == "__main__":
    parser = _command_line_parser()
    args = parser.parse_args()
    args = postprocess_args(args)

    corpus = get_lm_corpus(args.dataset_path, args.dataset, args.vocab_file, args.cased)

    # Use the actual number of tokens from dictionary
    ntokens = len(corpus.vocab)
    args.n_token = ntokens

    trainer = OnlineTrainer(
                args,
                corpus,
            )

    epoch = 0
    done = trainer.train(epoch)
    # remove epoch snapshot to save memory
    trainer.save_snapshot(-1)
    val_token_loss, val_word_loss = batch_evaluate(trainer.test_data, trainer.model, trainer.args)
    print("* Final Model Ends at Epoch #{}".format(epoch+1))
    print("| val token/word ppl {:9.3f} / {:9.3f} ".format(math.exp(val_token_loss), math.exp(val_word_loss)))

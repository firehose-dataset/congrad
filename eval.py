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

from collections import defaultdict

from core.dataset import get_lm_corpus
from core.configs import get_basic_parser
from core.inference import ModelWrapper, TOKENIZER
from core.models import MemTransformerLM, MTLMemTransformerLM
from core.trainer.trainer_util import *

from tqdm import tqdm

#TODO: Dangerous lines of code below, make sure remove it when you don't know what you ignored
import warnings
warnings.filterwarnings("ignore")

def postprocess_args(args):
    args.tie_weight = not args.not_tied
    args.d_embed = args.d_model if args.d_embed < 0 else args.d_embed
    assert args.batch_size % args.batch_chunk == 0
    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
            args.fp16 = False
        else:
            try:
                from apex.fp16_utils import FP16_Optimizer
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                args.fp16 = False

    return args

def _command_line_parser():
    parser = argparse.ArgumentParser(parents=[get_basic_parser()])
    parser.add_argument('--data', type=str, default='data/cikm2010_dataset',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cikm2010',
                        help='dataset name')
    parser.add_argument('--cased', default=False, action='store_true',
                        help='use cased or uncased corpus')
    parser.add_argument('--vocab_file', type=str, required=True,
                        help='vocabulary')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='model directory')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='data split to evaluate')
    parser.add_argument('--epoch', type=int, default=None,
                        help='model of epoch to load')
    return parser

def batch_evaluate(eval_data, model, args, verbose=False):
    model.eval()
    model.reset_length(280, 0)

    user_total_loss = defaultdict(lambda: int(0))
    user_word_len   = defaultdict(lambda: int(0))
    user_word_ppl   = {}
    # Evaluation
    total_word_len, total_token_len, total_loss = 0, 0, 0.
    with torch.no_grad():
        if verbose: eval_data = tqdm(eval_data, ncols=64)
        mems = tuple()
        for i, (data, target, user, token_len, word_len) in enumerate(eval_data):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, user, *mems)
            loss, mems = ret[0], ret[1:]
            total_loss += loss.sum().float().item()
            total_word_len += word_len.sum().item()
            total_token_len += token_len.sum().item()

            for ix, uid in enumerate(user.flatten().tolist()):
                user_total_loss[uid] += loss[:, ix].sum().float().item()
                user_word_len[uid]   += word_len[0, ix].item()

    # Switch back to the training mode
    model.train()
    model.reset_length(args.max_seqlen, 0)

    return total_loss / total_token_len, total_loss / total_word_len, \
           user_total_loss, user_word_len


if __name__ == '__main__':
    parser = _command_line_parser()
    args = parser.parse_args()
    postprocess_args(args)
    print("Evaluating {}".format(args.model_dir))

    corpus = get_lm_corpus(args.data, args.dataset, args.vocab_file, args.cased, use_bos=args.use_bos)

    # Use the actual number of tokens from dictionary
    ntokens = len(corpus.vocab)
    args.n_token = ntokens
    args.word_dir = args.model_dir

    device = torch.device('cuda' if args.cuda else 'cpu')
    model_filepath = os.path.join(args.model_dir, 'model.pt')
    model_dict = torch.load(model_filepath)

    args_filepath = os.path.join(args.model_dir, 'configs.json')
    args_json = json.load(open(args_filepath))
    saved_args = argparse.Namespace(**args_json)

    model, _ = prepare_model(saved_args, device)
    model.load_state_dict(model_dict)

    data_iter = corpus.get_iterator(args.split, args.batch_size, args.max_seqlen,
                        device=device, shuffle=False, subword_augment=args.subword_augment)

    # reload arguments (eval_tgt_len and eval_mem_len are very important variables)
    model.args = args
    model.same_length = args.same_length
    model.clamp_len = args.clamp_len

    print(' * start evaluating model...')
    token_loss, word_loss, user_total_loss, user_word_len = batch_evaluate(data_iter, model, args, verbose=True)
    print(' | {} token/word ppl {:9.3f} / {:9.3f} '.format(args.split, math.exp(token_loss), math.exp(word_loss)))

    print('Saving user word ppl files to: {}'.format(os.path.join(args.model_dir, 'user_wppl.pt')))
    torch.save({'total_loss': user_total_loss,
                'word_len': user_word_len },
                os.path.join(args.model_dir, 'user_rst.pt'))

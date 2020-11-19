import os
import time
import argparse

def get_basic_parser():
    parser = argparse.ArgumentParser(description='Transformer Language Model', add_help=False)
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                                help='batch size on gpu 0')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension')
    parser.add_argument('--d_user_embed', type=int, default=-1,
                        help='user_embedding dimension')
    parser.add_argument('--mtl_depth', type=int, default=1,
                        help='depth of the mtl model')
    parser.add_argument('--mtl_width', type=float, default=0.5,
                        help='width of the mtl model')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.001,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--parsimonious', action='store_true',
                        help='parsimonious mtl params')
    parser.add_argument('--max_step', type=int, default=50000,
                        help='upper step limit')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='the data chunk size. In continual learning, this is the rate of data arrival at each model learning step; In offline batch learning, this is the equivalent term to the mini-batch size of the model.')
    parser.add_argument('--eval_batch_size', type=int, default=60,
                        help='evaluation batch size')
    parser.add_argument('--eval_initial_model', action='store_true',
                        help='evaluate the initialized model')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    parser.add_argument('--max_seqlen', type=int, default=280,
                        help='number of tokens to predict')
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=5000,
                        help='evaluation interval')
    parser.add_argument('--work_dir', default='exp/LM', type=str,
                        help='experiment directory.')
    parser.add_argument('--model_class', type=str, default='MemTransformerLM',
                        choices=['MTLMemTransformerLM', 'MemTransformerLM'],
                        help='choose transformer model')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--break_ratio', default=1.0, type=float,
                        help='number of data to break down')
    parser.add_argument('--subword-augment', action='store_true',
                        help='perform subword augmentation')
    parser.add_argument('--mtl-type', type=str, default='layerwise',
                        choices=['multi_encoder', 'multi_decoder', 'layerwise', 'all'],
                        help='types of multitask learning architecture')
    parser.add_argument('--snapshot_dir', type=str, default=None,
                        help='resume snapshot dir')
    parser.add_argument('--init_weights', type=str, default=None,
                        help='weights to init ')
    parser.add_argument('--postfix', type=str, default=None,
                        help='postfix of experiment')
    parser.add_argument('--use_tb_logger', action='store_true',
                        help='Turn on tensorboard logger.')
    parser.add_argument('--async_lr', action='store_true',
                        help='Smaller lr for backbone and larger lr for mtl.')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from the saved checkpoint')
    parser.add_argument('--resume_dir', type=str, default='',
                        help='resume directory')
    parser.add_argument('--eval_online_fit', action='store_false',
                        help='Evaluate online loss as metric for online fit.')
    return parser

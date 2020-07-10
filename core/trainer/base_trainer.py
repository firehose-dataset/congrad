import time
import math
import os, sys
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.logger import Logger
from core.dataset import Corpus
from core.trainer.trainer_utils import (
    prepare_model,
    prepare_logger,
    prepare_optimizer,
    prepare_scheduler,
)
from core.models import MemTransformerLM, MTLMemTransformerLM

class BaseTrainer:
    """
    LM Trainer
    """
    def __init__(self,
        args,
        corpus: Corpus,
        scripts_to_save,
        ):

        # Specify device
        device = torch.device('cuda' if args.cuda else 'cpu')

        self.args = args
        self.device = device
        self.corpus = corpus
        self.logging = prepare_logger(args, scripts_to_save)

        self.args.n_user = len(corpus.user_dict)
        self.model, self.para_model = prepare_model(self.args, self.device)
        self.args.n_all_param = sum([p.nelement() for p in self.model.parameters()])
        self.args.n_nonemb_param  = sum([p.nelement() for p in self.model.layers.parameters()])
        self.args.n_nonemb_param += sum([p.nelement() for p in self.model.mtl_layers.parameters()] if isinstance(self.model, MTLMemTransformerLM) else [])

        self.optimizer = prepare_optimizer(self.args, self.model)
        self.scheduler = prepare_scheduler(self.args, self.optimizer)
        self.max_step = self.args.max_step

        self.train_step = 0
        self.train_loss = 0
        self.train_word_length = 0
        self.train_token_length = 0
        self.best_val_loss = None
        self.eval_start_time = time.time()
        self.log_start_time = time.time()

        # Finish starting up
        self.logging('=' * 100)
        for k, v in args.__dict__.items():
            self.logging('    - {} : {}'.format(k, v))
        self.logging('=' * 100)
        self.logging('#params = {}'.format(self.args.n_all_param))
        self.logging('#non emb params = {}'.format(self.args.n_nonemb_param))

        # Tensorboard Support
        self.logger = Logger(args, self.args.work_dir, flush_secs=10)
        if args.snapshot_dir is not None:
          self.load_snapshot()
        elif  args.init_weights is not None:
          self.load_weight(args.init_weights)
        else:
          print("[Warning] Random initialized model")

    def train(self, epoch):
        self.model.train()
        return self.try_training(epoch)

    def try_training(self, epoch):
        raise NotImplementedError('Trainer specific function not implemented:'
                + ' try_evaluate(self, epoch, batch)')

    def try_evaluate(self, epoch, batch):
        raise NotImplementedError('Trainer specific function not implemented:'
                + ' try_evaluate(self, epoch, batch)')

    def try_logging(self, epoch, batch):
        raise NotImplementedError('Trainer specific function not implemented:'
                + ' try_logging(self, epoch, batch)')

    def save_snapshot(self, epoch=None):
        """Save Model Snapshots"""
        model_filepath = os.path.join(self.args.work_dir, \
                                             'model.pt' if epoch is None \
                                        else 'model_epoch{:03d}.pt'.format(epoch))
        optimizer_filepath = os.path.join(self.args.work_dir, \
                                             'optimizer.pt' if epoch is None \
                                        else 'optimizer_epoch{:03d}.pt'.format(epoch))
        torch.save(self.model.state_dict(), model_filepath)
        torch.save(self.optimizer.state_dict(), optimizer_filepath)

    def load_snapshot(self, epoch=None):
        print('[*] Load weight and optimizer of epoch {}'.format('last' if epoch is None else '#{}'.format(epoch)))
        model_filepath = os.path.join(self.args.work_dir, \
                                             'model.pt' if epoch is None \
                                        else 'model_epoch{:03d}.pt'.format(epoch))
        optimizer_filepath = os.path.join(self.args.work_dir, \
                                             'optimizer.pt' if epoch is None \
                                        else 'optimizer_epoch{:03d}.pt'.format(epoch))
        self.load_weight(model_filepath)
        self.load_optimizer(optimizer_filepath)

    def load_optimizer(self, optimizer_filepath):
        self.optimizer.load_state_dict(torch.load(optimizer_filepath))

    def load_weight(self, model_filepath):
        loaded_state_dict = torch.load(model_filepath)
        current_state_dict = self.model.state_dict()
        if loaded_state_dict.keys() != current_state_dict.keys():
            print('[WARNING] Load only the partial model parameters...')
            loaded_state_dict = { k: v for k, v in loaded_state_dict.items() if k in current_state_dict }
            current_state_dict.update(loaded_state_dict)
            self.model.load_state_dict(current_state_dict)
        else:
            self.model.load_state_dict(loaded_state_dict)

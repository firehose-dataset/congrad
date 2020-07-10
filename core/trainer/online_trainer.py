import time
import json
import math
import os, sys
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core.algos import (
    AGEM,
    OnlineOnly,
    ReplayOnly,
    MixedReplay,
    ConGraD_AGEM,
    ConGraD_OnlineOnly,
    ConGraD_ReplayOnly,
    ConGraD_MixedReplay
)
from core.dataset import Corpus
from core.exp_utils import *
from core.trainer.base_trainer import BaseTrainer
from core.trainer.trainer_utils import (
    batch_evaluate,
    prepare_buffer,
    step_lr_scheduler,
)

# constants
MAX_TWEET_LENGTH = 280

class OnlineTrainer(BaseTrainer):
    def __init__(
            self,
            args,
            corpus: Corpus,
            scripts_to_save = None,
        ):
        assert args.scheduler in {'constant'}, \
            'Online learning assumes no knowledge about max_step and only constant scheduler is applicable'
        super().__init__(
                args,
                corpus,
                scripts_to_save,
            )

        self.stats = dict(
                num_data=0,                     # Number of data encountered so far
                num_updates=0,                  # Number of model updates so far
                train_step=0,                   # Global train counter
                online_loss=0,                  # Online Cumulative loss
                online_word_count=0,            # Online Cumulative word counter
                online_token_count=0,           # Online Cumulative token counter
                total_loss=0,                   # Cumulative loss
                total_word_count=0,             # Cumulative word counter
                total_token_count=0,            # Cumulative token counter
                best_val_loss=None,             # Best validation loss so far
                eval_start_time=time.time(),    # Time of last evaluation start
                log_start_time=time.time(),     # Time of last logging end
            )

        self.args = args
        self.eval_interval = args.eval_interval

        self.total_replay_rbsize = max( args.batch_size, args.replay_per_user_rbsize * corpus.num_users)
        self.total_online_rbsize = max( args.batch_size, args.online_per_user_rbsize * corpus.num_users)
        # prepare online cache and replay buffer
        self.online_buffer = prepare_buffer(
                                args.online_buffer_strategy,
                                args.online_per_user_rbsize,
                                self.total_online_rbsize,
                                args.max_seqlen,
                            )
        self.replay_buffer = prepare_buffer(
                                args.replay_buffer_strategy,
                                args.replay_per_user_rbsize,
                                self.total_replay_rbsize,
                                args.max_seqlen,
                            )

        params = ( self.model, self.para_model, self.optimizer,
                   self.online_buffer, self.replay_buffer, args )
        # setup learning algorithm
        self.learning_algorithm = eval(args.learner)(*params)

        assert (args.snapshot_dir is None) or \
               (args.learner in {'AGEMOnline', 'OMSER'}), 'Can not resume, something went wrong.'

        print('[*] Online Cache Size: {}, Replay Buffer Size: {}'.format(self.total_online_rbsize,
                                                                         self.total_replay_rbsize))
        print('[*] Learning Algorithm: {}'.format(str(self.learning_algorithm)))
        # train iterator should be online and must not be shuffled
        self.train_data = corpus.get_iterator(
                            'train',
                            args.batch_size,
                            args.max_seqlen,
                            iterator_name = 'LMOnePassIterator',
                            device = self.device,
                            subword_augment = args.subword_augment,
                            shuffle = False,
                        )

        # always load batch iterator in batch learning mode
        self.val_data = corpus.get_iterator(
                            'val',
                            args.eval_batch_size,
                            args.max_seqlen,
                            iterator_name = 'LMBatchIterator',
                            device = self.device,
                            shuffle = True,
                            subword_augment = False,
                        )

        self.test_data = corpus.get_iterator(
                            'test',
                            args.eval_batch_size,
                            args.max_seqlen,
                            iterator_name = 'LMBatchIterator',
                            device = self.device,
                            shuffle = True,
                            subword_augment = False,
                        )

        self.timer = Timer()

        self.resume_numdata = -1
        if args.snapshot_dir is not None:
            print('Resuming AGEMOnline from {}'.format(args.snapshot_dir))
            scalar_filepath = os.path.join(args.work_dir, 'scalars.json')
            with open(scalar_filepath, 'r') as in_file:
              scalars = json.load(in_file)
            self.resume_numdata = max([ int(_) for _ in scalars['Train/Num_Data'].values()])
            print('Skipping through {} data'.format(self.resume_numdata))

    def try_training(self, epoch):
        args = self.args
        assert args.batch_chunk == 1, 'This version requires batch chunk to be 1.'

        if args.eval_initial_model: self.evaluate(0, 0)
        for batch_id, online_batch in enumerate(self.train_data):
            # (1) Learning: distribute the learning job for one step
            self.learning_algorithm(online_batch, self.stats,
                skip_optim=(self.stats['num_data'] < self.resume_numdata))

            # (2) Postprocess: update global stat tracker
            self.stats['train_step'] += 1
            self.stats['num_data'] += len(online_batch[0])
            self.try_logging(epoch, batch_id)

            # (3) Evaluation (if it is scheduled): evaluate model
            val_loss = self.try_evaluate(epoch, batch_id)
            step_lr_scheduler(self.args, self.scheduler, self.optimizer,
                                         self.stats['train_step'], val_loss)

        return False

    def evaluate(self, epoch, batch_id):
        self.stats['eval_start_time'] = time.time()
        # batch evaluate
        val_token_loss, val_word_loss = batch_evaluate(self.val_data, self.model, self.args)
        self.logging('-' * 125)
        log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s | val loss {:5.2f}'.format(
                        self.stats['train_step'] // self.eval_interval,
                        self.stats['train_step'],
                        (time.time() - self.stats['eval_start_time']),
                        val_word_loss
                    )
        log_str += ' | val token/word ppl {:9.3f} / {:9.3f} '.format(
                        math.exp(val_token_loss),
                        math.exp(val_word_loss)
                    )
        self.logging(log_str)
        self.logging('-' * 125)

        # Save the model with best validation loss
        if not self.stats['best_val_loss'] or val_word_loss < self.stats['best_val_loss']:
            self.save_snapshot()
            self.stats['best_val_loss'] = val_word_loss

        self.logger.add_scalar('Validate/Token_Perplexity',
                                    math.exp(val_token_loss),
                                    self.stats['train_step'])
        self.logger.add_scalar('Validate/Word_Perplexity',
                                    math.exp(val_word_loss),
                                    self.stats['train_step'])

        return val_token_loss, val_word_loss

    def try_evaluate(self, epoch, batch_id):
        val_token_loss = None
        val_word_loss = None
        if (self.stats['train_step'] % self.eval_interval == 0) and (self.stats['num_data'] > self.resume_numdata):
          val_token_loss, val_word_loss = self.evaluate(epoch, batch_id)
        return val_word_loss

    def try_logging(self, epoch, batch_id):
        if self.stats['train_step'] % self.args.log_interval == 0:
            train_word_loss = self.stats['total_loss']      \
                            / max(self.stats['total_word_count'], 1e-10)
            train_token_loss = self.stats['total_loss']     \
                             / max(self.stats['total_token_count'], 1e-10)
            online_word_loss = self.stats['online_loss']   \
                             / max(self.stats['online_word_count'], 1e-10)
            online_token_loss = self.stats['online_loss'] \
                              / max(self.stats['online_token_count'], 1e-10)

            elapsed = time.time() - self.stats['log_start_time']
            log_str = '| global timer {} | epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.2g} ' \
                      '| ms/batch {:5.2f} | total loss {:5.2f} '   \
                      '| online loss {:5.2f}'.format(
                            self.timer.measure(),
                            epoch,
                            self.stats['train_step'],
                            batch_id + 1,
                            self.optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / self.args.log_interval,
                            train_token_loss,
                            online_token_loss,
                        )
            log_str += ' | online token ppl {:9.3f}'.format(
                            math.exp(online_token_loss)
                        )
            self.logging(log_str)
            self.logger.add_scalar('Online/Token_Perplexity',
                                    math.exp(online_token_loss),
                                    self.stats['train_step'])

            self.logger.add_scalar('Online/Word_Perplexity',
                                      math.exp(online_word_loss),
                                      self.stats['train_step'])

            self.logger.add_scalar('Train/Token_Perplexity',
                                      math.exp(train_token_loss),
                                      self.stats['train_step'])

            self.logger.add_scalar('Train/Word_Perplexity',
                                      math.exp(train_word_loss),
                                      self.stats['train_step'])

            self.logger.add_scalar('Train/Num_Updates',
                                      self.stats['num_updates'],
                                      self.stats['train_step'])

            self.logger.add_scalar('Train/Num_Data',
                                      self.stats['num_data'],
                                      self.stats['train_step'])

            self.logger.dump()

            # reset stats
            self.stats['total_loss'] = 0
            self.stats['total_word_count'] = 0
            self.stats['total_token_count'] = 0
            self.stats['online_loss'] = 0
            self.stats['online_word_count'] = 0
            self.stats['online_token_count'] = 0
            self.stats['log_start_time'] = time.time()


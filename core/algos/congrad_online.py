import torch
import itertools
import numpy as np
from torch.nn.utils import clip_grad_norm_

from core.algos.base import Algorithm
from core.dataset.utils import encap_batch

class ConGraD_OnlineOnly(Algorithm):
    def __init__(
        self,
        model,
        para_model,
        optimizer,
        online_buffer,
        replay_buffer,
        args,
    ):
        assert args.online_buffer_strategy in {'greedy', 'stratified', 'reservoir'}
        assert args.replay_buffer_strategy in {'greedy', 'stratified', 'reservoir'}

        self.clip_value = args.clip
        self.device = next(model.parameters()).device
        self.eval_online_fit = args.eval_online_fit

        self.model = model
        self.para_model = para_model
        self.optimizer = optimizer

        self.max_k_steps = args.max_k_steps
        self.replay_buffer = replay_buffer
        self.online_buffer = online_buffer
        self.replay_bsz = args.replay_batch_size
        self.online_bsz = args.online_batch_size

        self.allow_zero_step = args.allow_zero_step

        self.param_groups = self.optimizer.param_groups
        self.score_banks = [ np.inf for i in range(args.max_k_steps + 1) ]
        self.model_banks = [ [ [ p.clone().detach() for p in group['params'] ] \
                                              for group in self.param_groups ] \
                                              for i in range(args.max_k_steps + 1) ]
        for model_bank in self.model_banks:
            for w in itertools.chain(*model_bank):
                w.requires_grad = False

    def take_snapshot(self, k, score):
        # store model scores
        self.score_banks[k] = score
        # store snapshot weights
        for snapshot_weights, model_weights in zip(self.model_banks[k], self.param_groups):
            for snapshot_weight, model_weight in zip(snapshot_weights, model_weights['params']):
                snapshot_weight.data.copy_(model_weight.data)

    def resume_snapshot(self):
        best_k = np.argmin(self.score_banks)
        for snapshot_weights, model_weights in zip(self.model_banks[best_k], self.param_groups):
            for snapshot_weight, model_weight in zip(snapshot_weights, model_weights['params']):
                model_weight.data.copy_(snapshot_weight.data)

        return best_k

    def forward_eval(self, data_encodes):
        self.model.eval()
        with torch.no_grad():
            data, target, users, token_len, word_len = data_encodes[:5]
            online_loss = self.para_model(data, target, users, *tuple())[0]
        self.model.train()

        return online_loss, token_len, word_len

    def forward(self, online_batch, stats, skip_optim=False):
        mems = tuple()
        if not self.eval_online_fit:
            stats['online_loss']        += 0
            stats['online_word_count']  += 1e-20
            stats['online_token_count'] += 1e-20
        else:
            # Evaluate online batch.
            total_loss, token_len, word_len = self.forward_eval(
                encap_batch(online_batch, max([ len(_) for _ in online_batch[0] ]), self.device),
            )

            stats['online_loss']        += total_loss.float().sum().item()
            stats['online_word_count']  += word_len.float().sum().item()
            stats['online_token_count'] += token_len.float().sum().item()

        # Update buffer: add online data stream into replay buffer
        retired_batch = self.online_buffer.add_batch(*online_batch)
        if retired_batch is not None: self.replay_buffer.add_batch(*retired_batch)

        # Start training only after replay buffer has data
        if self.replay_buffer.sample_batch(self.replay_bsz, self.device) is None: return

        # Sample a cached (validation) batch for model selection
        online_encodes = self.online_buffer.sample_batch(
                            self.online_bsz, self.device,
                        )

        # Snapshot Current Model
        if self.allow_zero_step:
            val_loss, _, _ = self.forward_eval(online_encodes)
            self.take_snapshot(0, val_loss.sum().item())

        # * Optimize for k step on current dataset
        for k in range(1, self.max_k_steps+1):
            # * Sample a batch of data from replay buffer
            replay_encodes = self.replay_buffer.sample_batch(
                                self.replay_bsz,
                                self.device,
                            )

            # decapsulate batch
            data, target, users, token_len, word_len = replay_encodes[:5]
            raw_loss = self.para_model(data, target, users, *mems)[0]
            replay_psl = raw_loss.sum(dim=0, keepdim=True) \
                       / token_len.type_as(raw_loss)

            replay_loss = replay_psl.mean()

            self.optimizer.zero_grad()
            replay_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

            # Take snapshot
            val_loss, _, _ = self.forward_eval(online_encodes)
            self.take_snapshot(k, val_loss.sum().item())

            # * Update global stats
            stats['num_updates']        += 1
            stats['total_loss']         += raw_loss.sum().float().item()
            stats['total_word_count']   += word_len.sum().float().item()
            stats['total_token_count']  += token_len.sum().float().item()

        # choose best snapshot based on current online measure
        best_k = self.resume_snapshot()
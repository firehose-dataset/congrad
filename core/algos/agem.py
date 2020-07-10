import os
import json
import torch
import itertools
from torch.nn.utils import clip_grad_norm_

from core.algos.base import Algorithm
from core.dataset.utils import encap_batch

def extract_grads(params, grad, grad_dims):
    grad.fill_(0)
    cnt = 0
    for param in params:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            end = sum(grad_dims[:cnt + 1])
            grad[beg:end].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(params, newgrad, grad_dims):
    cnt = 0
    for param in params:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

class AGEM(Algorithm):
    """ Pytorch implementation of averaged gradient episodic memory """
    def __init__(self, model, para_model, optimizer,
                 online_buffer, replay_buffer, args):
        assert args.online_buffer_strategy in {'greedy', 'stratified', 'reservoir'}
        assert args.replay_buffer_strategy in {'greedy', 'stratified', 'reservoir'}

        self.eval_online_fit = args.eval_online_fit # always use new data or not
        self.clip_value = args.clip
        self.device = next(model.parameters()).device

        self.model = model
        self.para_model = para_model
        self.optimizer = optimizer
        self.max_k_steps = args.max_k_steps

        self.online_buffer = online_buffer
        self.replay_buffer = replay_buffer
        self.online_bsz = args.online_batch_size
        self.replay_bsz = args.replay_batch_size

        self.grad_dims = []
        # different parameter numbers
        model_params = itertools.chain(*[
                            model.word_emb.parameters(),
                            model.layers.parameters(),
                            model.mtl_layers.parameters(),
                        ])

        for param in model_params:
            self.grad_dims.append(param.data.numel())

        self.replay_grad = torch.zeros(sum(self.grad_dims)).to(self.device)
        self.online_grad = torch.zeros(sum(self.grad_dims)).to(self.device)

    def forward_eval(self, data_encodes):
        self.model.eval()
        with torch.no_grad():
            data, target, users, token_len, word_len = data_encodes[:5]
            online_loss = self.para_model(data, target, users, *tuple())[0]
        self.model.train()

        return online_loss, token_len, word_len

    def forward(self, online_batch, stats, skip_optim=False):
        # Update buffer: add online data stream into replay buffer
        retired_batch = self.online_buffer.add_batch(*online_batch)
        if retired_batch is not None: self.replay_buffer.add_batch(*retired_batch)

        # Start training only after replay buffer has data
        if self.replay_buffer.sample_batch(self.replay_bsz, self.device) is None:
            return

        # Skip optimization if in resume mode
        if skip_optim: return

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

        mems = tuple()
        model_params = list(itertools.chain(*[
                            self.model.word_emb.parameters(),
                            self.model.layers.parameters(),
                            self.model.mtl_layers.parameters(),
                        ]))

        # * Optimize for k step on current dataset
        for k in range(self.max_k_steps):
            # Optimization on Replay Data
            replay_encodes = self.replay_buffer.sample_batch(
                                self.replay_bsz, self.device,
                            )

            data, target, users, token_len, word_len = replay_encodes[:5]
            total_loss = self.para_model(data, target, users, *mems)[0]

            replay_psl  = total_loss.sum(dim=0, keepdim=True) / token_len.type_as(total_loss)
            replay_loss = replay_psl.mean()

            # (*) Extract Replay Gradients
            self.optimizer.zero_grad()
            replay_loss.backward()
            extract_grads(model_params, self.replay_grad, self.grad_dims)

            # * Sample a batch of data from replay buffer
            online_encodes = self.online_buffer.sample_batch(
                                self.online_bsz, self.device,
                            )

            # decapsulate batch
            data, target, users, token_len, word_len = online_encodes[:5]
            total_loss = self.para_model(data, target, users, *mems)[0]

            online_psl  = total_loss.sum(dim=0, keepdim=True) / token_len.type_as(total_loss)
            online_loss  = online_psl.mean()

            # (*) Extract Online Gradients
            self.optimizer.zero_grad()
            online_loss.backward()
            extract_grads(model_params, self.online_grad, self.grad_dims)

            online_replay_dp = torch.mm(self.online_grad.view(1, -1), self.replay_grad.view(-1, 1))
            if online_replay_dp.item() < 0:
                # constraint violation, next batch
                # reproject gradient when the constraint is violated
                replay_replay_dp = torch.mm(self.replay_grad.view(1, -1),
                                            self.replay_grad.view(-1, 1))
                self.online_grad.copy_( self.online_grad -  \
                            (online_replay_dp.item() / replay_replay_dp.item()) * self.replay_grad)
                overwrite_grad(model_params, self.online_grad, self.grad_dims)

            # clip gradient norm
            clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.optimizer.step()

            # * Update global stats
            stats['num_updates']        += 1
            stats['total_loss']         += total_loss.sum().float().item()
            stats['total_word_count']   += word_len.sum().float().item()
            stats['total_token_count']  += token_len.sum().float().item()

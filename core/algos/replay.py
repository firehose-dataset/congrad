import torch
from torch.nn.utils import clip_grad_norm_

from core.algos.base import Algorithm
from core.dataset.utils import encap_batch

class ReplayOnly(Algorithm):
    def __init__(
        self,
        model,
        para_model,
        optimizer,
        online_buffer,
        replay_buffer,
        args,
    ):
        assert args.online_buffer_strategy in {'greedy'}
        assert args.replay_buffer_strategy in {'greedy', 'stratified', 'reservoir'}

        self.clip_value = args.clip
        self.device = next(model.parameters()).device

        self.model = model
        self.para_model = para_model
        self.optimizer = optimizer

        self.max_k_steps = args.max_k_steps
        self.replay_buffer = replay_buffer
        self.online_buffer = online_buffer
        self.replay_bsz = args.replay_batch_size
        self.online_bsz = args.online_batch_size

        self.eval_online_fit = args.eval_online_fit

    def forward_eval(self, data_encodes):
        self.model.eval()
        with torch.no_grad():
            data, target, users, token_len, word_len = data_encodes[:5]
            online_loss = self.para_model(data, target, users, *tuple())[0]
        self.model.train()

        return online_loss, token_len, word_len

    def forward(self, online_batch, stats, skip_optim=False):
        mems = tuple()
        # Evaluate online batch.
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
        if self.replay_buffer.sample_batch(self.replay_bsz, self.device) is None:
            return

        # * Optimize for k step on current dataset
        for k in range(self.max_k_steps):
            # * Sample a batch of data from replay buffer
            replay_encodes = self.replay_buffer.sample_batch(
                                self.replay_bsz, self.device,
                            )

            # reset optimizer gradients
            self.optimizer.zero_grad()

            data, target, users, token_len, word_len, weights = replay_encodes[:6]
            raw_loss = self.para_model(data, target, users, *mems)[0]
            per_sample_loss = raw_loss.float().sum(dim=0, keepdim=True) \
                            / token_len.type_as(raw_loss)

            total_loss = (per_sample_loss * weights).mean()
            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_value)

            self.optimizer.step()

            # * Update global stats
            stats['num_updates']        += 1
            stats['total_loss']         += raw_loss.float().sum().item()
            stats['total_word_count']   += word_len.float().sum().item()
            stats['total_token_count']  += token_len.float().sum().item()

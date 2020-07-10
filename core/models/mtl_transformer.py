import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.embeddings import AdaptiveEmbedding, PositionalEmbedding
from core.models.mem_transformer import (
    MemTransformerLM
)

def identity(x):
    return x

class MTLMemTransformerLM(MemTransformerLM):
    def __init__(self,
            *args,
            mtl_type='multi_decoder',
            d_user_embed=-1,
            mtl_width=4,
            mtl_depth=1,
            **kwargs,
        ):
        super().__init__(
              *args,
              **kwargs,
            )
        assert hasattr(self, 'n_user'), 'num of users must be specified'

        self.mtl_type = mtl_type
        assert d_user_embed != -1, '[Error] d_user_embed must be specified.'
        self.d_user_embed = d_user_embed
        self.user_emb  = AdaptiveEmbedding(self.n_user, self.d_user_embed, self.d_model)

        self.num_mtl_layers = 0
        self.mtl_layers = nn.ModuleList()
        if self.mtl_type in { 'multi_encoder', 'multi_decoder' }:
            self.num_mtl_layers = 1
        elif self.mtl_type in { 'layerwise' }:
            self.num_mtl_layers = len(self.layers)
        elif self.mtl_type in { 'all' }:
            self.num_mtl_layers = len(self.layers) + 2
        else:
            raise ValueError('Unobserved Value.')

        def build_mtl_layer(d_model, width, depth, dropout):
            d_inner = int(d_model * width)
            mtl_layer = [ nn.Linear(2*d_model, d_inner), nn.ReLU(inplace=True) ]
            for _ in range(depth - 1):
              mtl_layer.extend([ nn.Linear(d_inner, d_inner), nn.ReLU(inplace=True) ])
            mtl_layer.extend([ nn.Dropout(dropout), nn.Linear(d_inner, d_model), nn.Dropout(dropout), nn.LayerNorm(d_model) ])
            return mtl_layer

        for _ in range(self.num_mtl_layers):
            self.mtl_layers.append(
                nn.Sequential(
                    *build_mtl_layer(self.d_model, mtl_width, mtl_depth, self.dropout),
                )
            )

    def forward(self, data, target, user, *mems):
        data, target = self.preprocess_data(data, target)
        tgt_len = target.size(0)

        logit, new_mems = self.forward_model(data, user, mems, tgt_len=tgt_len)
        logit  = logit.view(-1, logit.size(-1))
        target = target.view(-1)

        nll  = F.nll_loss(
                    F.log_softmax(logit, dim=-1),
                    target,
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )
        loss = nll.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

    def _forward(self, word_emb, dec_inp, user_emb, mems=None):
        qlen, bsz = word_emb.size(0), word_emb.size(1)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        all_ones = word_emb.new_ones(qlen, klen)
        dec_attn_mask = torch.triu(
            all_ones, diagonal=1+mlen).byte()[:,:,None]

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            if self.mtl_type in {'layerwise'}:
                task_specific = self.mtl_layers[i](torch.cat([core_out, user_emb.expand_as(core_out)], dim=-1))
                core_out = core_out + task_specific
            elif self.mtl_type in {'all'}: # since all layers includes an additional layer
                task_specific = self.mtl_layers[i+1](torch.cat([core_out, user_emb.expand_as(core_out)], dim=-1))
                core_out = core_out + task_specific
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(dec_inp, hids, mems, mlen, qlen)
        return core_out, new_mems

    def forward_model(self, data, user, mems, tgt_len=1):
        if not mems: mems = self.init_mems(data.size(1))

        user_emb = self.user_emb(user)
        word_emb = self.word_emb(data)

        # incorporate multitask information early
        if self.mtl_type in {'multi_encoder', 'all'}:
            task_specific = self.mtl_layers[0](torch.cat([word_emb, user_emb.expand_as(word_emb)], dim=-1))
            word_emb = word_emb + task_specific

        # shared transformer step
        hidden, new_mems = self._forward(word_emb, data, user_emb, mems=mems)
        pred_hid = hidden[-tgt_len:]

        # incorporate multitask information late
        if self.mtl_type in {'multi_decoder', 'all'}:
            task_specific = self.mtl_layers[-1](torch.cat([pred_hid, user_emb.expand_as(pred_hid)], dim=-1))
            pred_hid = pred_hid + task_specific

        return self.out_layer(pred_hid), new_mems

    def get_logprob(self, data, user, *mems, temperature=1.0):
        data, _ = self.preprocess_data(data, None)
        logit, _ = self.forward_model(data, user, mems, tgt_len=data.size(0))

        return F.log_softmax(logit / temperature, dim=-1)[:, :, self.token_offset:] # get rid of the first padding token

if __name__ == '__main__':
    import argparse
    import sys
    import os
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from core.corpus.iterators import LMShuffledIterator
    from ipdb import launch_ipdb_on_exception

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 32
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 16000
    args.n_user = 10

    data = [ np.random.randint(0, args.n_token, size=(data_len)).tolist() + [-1] * 20 for _ in range(B*10) ]
    user = [ np.random.randint(0, args.n_user) for _ in range(B*10) ]
    diter = LMShuffledIterator(data, user, B, tgt_len, ext_len=ext_len, device=device)

    with launch_ipdb_on_exception():
        for d_embed in [200, 100]:
            model = MTLMemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            dropatt=args.dropout, tie_weight=True, same_length=False,
                            d_embed=d_embed, pre_lnorm=True).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            print('Start new iterators')
            for idx, (inp, tgt, users, seq_len) in enumerate(diter):
                print('batch {}'.format(idx))
                loss = model(inp, tgt, users)
                from IPython import embed; embed()

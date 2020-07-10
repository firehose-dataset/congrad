import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.decoder import RelPartialLearnableDecoderLayer
from core.models.embeddings import AdaptiveEmbedding, PositionalEmbedding
from core.constants import constants

class MemTransformerLM(nn.Module):
    def __init__(self,
            n_user,
            n_token,
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            dropout,
            dropatt,
            *args,
            tgt_len=None,
            ext_len=None,
            mem_len=None,
            tie_weight=True,
            d_embed=None,
            clamp_len=-1,
            ignore_index=0,
            **kwargs,
        ):

        super(MemTransformerLM, self).__init__()
        # nll loss related
        if ignore_index != 0: print('Ignore index is deprecated...')
        self.ignore_index = constants.MODEL_PAD_INDEX
        self.token_offset = constants.MODEL_INDEX_OFFSET
        self.reduction = 'none'

        self.n_user = n_user
        self.n_token = n_token + self.token_offset

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(self.n_token, d_embed, d_model)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + mem_len

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt)
            )

        # use standard softmax
        self.out_layer = nn.Linear(d_model, self.n_token)
        if tie_weight:
            if d_model == d_embed:
                self.out_layer.weight = self.word_emb.emb_layers[0].weight
            else:
                print("[Warning] Dimensionality Mismatch Between WordEmb and Linear Classifier. Can't tie this weight...")
                pass
        self.tie_weight = tie_weight
        self.clamp_len = clamp_len

        print('Initializing {} with {} Users and {} Tokens (Padding Token=0)...'.format(
                  self.__class__.__name__, self.n_user, self.n_token))
        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len

    def init_mems(self, batch_size=0):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                if batch_size > 0:
                    mem = torch.zeros((self.mem_len, batch_size, self.d_model), dtype=param.dtype, device=param.device)
                else:
                    mem = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(mem)

            return mems
        else:
            return None

    def _update_mems(self, inps, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                ctx = cat[beg_idx:end_idx].detach()
                ctx.masked_fill_( (inps[-1, :] == constants.MODEL_PAD_INDEX)[None, :, None], 0)
                new_mems.append(ctx)

        return new_mems

    def preprocess_data(self, data, target):
        #IMPORTANT:
        #   - Shifting data and label by 1 (St. Pad Token index (which was set to be -1) becomes 0)
        data += self.token_offset
        if target is not None: target += self.token_offset
        return data, target

    # Transformer Forward:
    #   - Forward function that process all transformer related operations
    #   - Memeory related operation is also done here (dec_inp required to discard hiddens)
    def _forward(self, word_emb, dec_inp, mems=None):
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
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(dec_inp, hids, mems, mlen, qlen)
        return core_out, new_mems

    def forward(self, data, target, user, *mems):
        data, target = self.preprocess_data(data, target)

        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems(data.size(1))

        tgt_len = target.size(0)

        # extract word embedding
        word_emb = self.word_emb(data)
        hidden, new_mems = self._forward(word_emb, data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        logit  = self.out_layer(pred_hid)

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

    def get_logprob(self, data, user, *mems, temperature=1.0):
        data, _ = self.preprocess_data(data, None)
        if not mems: mems = self.init_mems(data.size(1))

        word_emb = self.word_emb(data)
        hidden, _ = self._forward(word_emb, data, mems=mems)
        logit  = self.out_layer(hidden)

        return F.log_softmax(logit / temperature, dim=-1)[:, :, self.token_offset:] # get rid of the first padding token

if __name__ == '__main__':
    import argparse
    import sys
    import os
    from IPython import embed
    import os.path as osp

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from core.iterators import LMShuffledIterator
    from core.dataset.corpus import get_lm_corpus
    from ipdb import launch_ipdb_on_exception

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--dropatt', type=float, default=0.5, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    # Load dataset
    parser.add_argument('--datadir', type=str, default='data/cikm2010_dataset',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cikm2010_freq',
                        choices=['cikm2010_freq'],
                        help='dataset name')
    parser.add_argument('--vocab_file', type=str, default='vocab/cikm2010_uncased_16000.model',
                        help='location of the vocabulary of the corpus')
    parser.add_argument('--cased', default=False, action='store_true',
                        help='whether to use cased or uncased corpus.')

    args = parser.parse_args()

    vocab_file = osp.abspath(args.vocab_file)
    print(vocab_file)
    corpus = get_lm_corpus(args.datadir, args.dataset, vocab_file, args.cased)
    print('Vocab size : {}'.format(len(corpus.vocab)))
    iterator = corpus.get_iterator('train', 512, 16, shuffle=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    B = 32
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = len(corpus.vocab)
    args.n_user = len(corpus.user_dict)

    with launch_ipdb_on_exception():
        for d_embed in [200, 100]:
            model = MemTransformerLM(
                            args.n_user, args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout, args.dropatt,
                            tie_weight=True, d_embed=d_embed).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            print('Start new iterators')
            for idx, ( inp, tgt, users, token_len, word_len ) in enumerate(iterator):
                print('batch {}'.format(idx))
                loss = model(inp, tgt, users)
                embed()
                break

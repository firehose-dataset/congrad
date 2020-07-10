import torch
import torch.nn as nn
import torch.optim as optim

from core.exp_utils import create_exp_dir
from core.dataset.utils import BalancedDataParallel
from core.models import MemTransformerLM, MTLMemTransformerLM
from core.memories import GreedyBuffer, ReservoirBuffer, StratifiedBuffer
### Batch Eval 
def batch_evaluate(eval_data, model, args, verbose=False):
    model.eval()
    model.reset_length(280, 0)

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

    # Switch back to the training mode
    model.train()
    model.reset_length(args.max_seqlen, 0)

    return total_loss / total_token_len, total_loss / total_word_len

#### buffer
def prepare_buffer(buffer_strategy, per_user_buffer_size, total_buffer_size, max_seqlen):
    memory_buffer = None
    # setup replay buffer
    if buffer_strategy == 'greedy':
        memory_buffer = GreedyBuffer(total_buffer_size,
                                     max_seqlen=max_seqlen)
    elif buffer_strategy == 'reservoir':
        memory_buffer = ReservoirBuffer(total_buffer_size,
                                        max_seqlen=max_seqlen)
    elif buffer_strategy == 'stratified':
        memory_buffer = StratifiedBuffer(per_user_buffer_size,
                                         buffer_class='GreedyBuffer',
                                         max_seqlen=max_seqlen)
    elif buffer_strategy == 'stratified-reservoir':
        memory_buffer = StratifiedBuffer(per_user_buffer_size,
                                         buffer_class='ReservoirBuffer',
                                         max_seqlen=max_seqlen)
    else:
        raise ValueError('Unsupported Strategy: {}.'.format(buffer_strategy))

    return memory_buffer

#### logger
def prepare_logger(args, scripts_to_save=None):
    logging = create_exp_dir(args.work_dir,
                scripts_to_save=scripts_to_save,
                debug=args.debug
            )
    return logging

#### scheduler
def prepare_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                       else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'constant':
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            return 1.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

def step_lr_scheduler(args, scheduler, optimizer, train_step, val_loss=None):
    if args.scheduler in ['cosine', 'constant', 'dev_perf']:
        # linear warmup stage
        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == 'cosine':
                scheduler.step(train_step)
            elif args.scheduler == 'dev_perf' and val_loss is not None:
                scheduler.step(val_loss)
    elif args.scheduler == 'inv_sqrt':
        scheduler.step(train_step)

#### model
def prepare_model(args, device):
    ###############################################################################
    # init methods (takes args locally)
    ###############################################################################
    def init_weight(weight):
        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = args.dropout

    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = args.dropatt

    if args.resume:
        assert args.resume_dir, "Resume directory must be non-empty."
        with open(os.path.join(args.resume_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not args.fp16:
            model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        model_class = eval(args.model_class)
        model = model_class(
                        args.n_user,
                        args.n_token,
                        args.n_layer,
                        args.n_head,
                        args.d_model,
                        args.d_head,
                        args.d_inner,
                        args.dropout,
                        args.dropatt,
                        tgt_len=args.max_seqlen,
                        ext_len=0,                  # no extension
                        mem_len=0,                  # no memory
                        mtl_type=args.mtl_type,
                        tie_weight=args.tie_weight,
                        d_embed=args.d_embed,
                        d_user_embed=args.d_user_embed,
                        mtl_width=args.mtl_width,
                        mtl_depth=args.mtl_depth,
                        clamp_len=args.clamp_len)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

    if args.multi_gpu:
        model = model.to(device)
        if args.gpu0_bsz >= 0:
            para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                              model, dim=1).to(device)
        else:
            para_model = nn.DataParallel(model, dim=1).to(device)
    else:
        para_model = model.to(device)

    return model, para_model

#### optimizer
def prepare_optimizer(args, model):
    if isinstance(model, MTLMemTransformerLM) and args.async_lr:
        params = [
              { 'params': model.word_emb.parameters(), 'lr': args.lr / 10.0},
              { 'params': model.layers.parameters(), 'lr': args.lr / 10.0},
              { 'params': model.user_emb.parameters(), 'lr': args.lr},
              { 'params': model.mtl_layers.parameters(), 'lr': args.lr},
            ]
    else:
        params = model.parameters()

    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr,
                momentum=args.mom)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr)

    return optimizer


"""One fixed, unpacked, equally example-weighted SFT pass. No model downloads."""
from contextlib import nullcontext

from experiment import recovery_prompt
from . import data

RECIPE = {
    'seed': 2001, 'epochs': 1, 'examples': 32, 'microbatch': 1, 'accumulation': 4,
    'trainable': 'last Llama decoder block self_attn q_proj and v_proj weights',
    'base_dtype': 'bfloat16', 'trainable_optimizer_dtype': 'float32', 'attention': 'sdpa',
    'optimizer': 'AdamW', 'lr': 0.0001, 'betas': [0.9, 0.999], 'eps': 1e-8,
    'weight_decay': 0., 'schedule': 'constant', 'clip_global_norm': 1., 'sequence_limit': 2048,
    'packing': False, 'cache': False, 'gradient_checkpointing': False,
    'loss': 'per-example mean shifted final answer and eot NLL',
}
UPDATES = RECIPE['examples'] // (RECIPE['accumulation'] * RECIPE['microbatch'])
BOS, EOT, PAD = 128000, 128009, 128004
IGNORE = -100


def tokenizer(path):
    from tokenizers import Tokenizer
    from experiment.reader_assets import model_lock
    if data.task.hash_file(path) != model_lock()['tokenizer.json']['sha256']:
        raise ValueError('Wrong tokenizer bytes')
    tok = Tokenizer.from_file(str(path))
    if [tok.token_to_id(s) for s in (recovery_prompt.BOS, '<|eot_id|>', '<|finetune_right_pad_id|>')] != [BOS, EOT, PAD]:
        raise ValueError('Unexpected special token IDs')
    return tok


def encode(example, tok):
    prompt = recovery_prompt.render(example['messages'], recovery_prompt.tool_definitions())
    suffix = example['target'] + '<|eot_id|>'
    def ids(text):
        return tok.encode(text, add_special_tokens=False).ids
    context, target = ids(prompt), ids(suffix)
    tokens = ids(prompt + suffix)
    if (tokens != context + target or tokens.count(BOS) != 1 or tokens[0] != BOS
            or target[-1] != EOT or target.count(EOT) != 1 or len(tokens) > RECIPE['sequence_limit']):
        raise ValueError('Invalid prompt/target token boundary, termination or length')
    return tokens, [IGNORE] * len(context) + target


def collate(examples, device):
    import torch
    width = max(len(ids) for ids, _ in examples)
    tokens, labels, attention = [], [], []
    for ids, target in examples:
        pad = width - len(ids)
        tokens.append(ids + [PAD] * pad)
        labels.append(target + [IGNORE] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return tuple(torch.tensor(rows, device=device) for rows in (tokens, labels, attention))


def loss(model, batch):
    import torch
    import torch.nn.functional as F
    tokens, labels, attention = batch
    # Logit at t predicts label at t+1. Project only positions with a target
    # in some row, keeping the entire attended context in the decoder.
    shifted = labels[:, 1:]
    positions = torch.where(shifted.ne(IGNORE).any(dim=0))[0]
    targets = shifted[:, positions]
    logits = model(tokens, attention_mask=attention, use_cache=False, logits_to_keep=positions).logits
    nll = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction='none', ignore_index=IGNORE)
    return (nll.sum(1) / targets.ne(IGNORE).sum(1)).mean()


def projections(model):
    block = len(model.model.layers) - 1
    params = dict(model.named_parameters())
    return {name: params[name] for kind in ('q_proj', 'v_proj')
            for name in [f'model.layers.{block}.self_attn.{kind}.weight']}


def fit(model, tok, device):
    import torch
    torch.manual_seed(RECIPE['seed'])
    model.requires_grad_(False)
    trainable = list(projections(model).values())
    for param in trainable:
        param.data = param.data.float()
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=RECIPE['lr'], betas=tuple(RECIPE['betas']),
                                  eps=RECIPE['eps'], weight_decay=RECIPE['weight_decay'], foreach=False)
    examples = [encode(x, tok) for x in data.load()]
    if len(examples) != RECIPE['examples']:
        raise ValueError('Incomplete SFT epoch')
    model.train()
    norms, losses = [], []
    for i, example in enumerate(examples, 1):
        with (torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()):
            value = loss(model, collate([example], device))
        (value / RECIPE['accumulation']).backward()
        losses.append(value.detach().item())
        if i % RECIPE['accumulation'] == 0:
            norm = torch.nn.utils.clip_grad_norm_(trainable, RECIPE['clip_global_norm'], error_if_nonfinite=True)
            optimizer.step()
            norms.append(norm.item())
            optimizer.zero_grad(set_to_none=True)
    return {'examples': len(examples), 'updates': len(norms), 'preclip_norms': norms, 'example_losses': losses,
            'supervised_tokens': sum(sum(t != IGNORE for t in labels) for _, labels in examples)}

"""CPU replay of the pinned synchronous loader and fit loop; no rollout or model."""
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from experiment import matched_contract as contract
from pinned_training_support import load


def replay(train_path, dev_path, seed, steps, shuffle=True):
    loaded = load('agentlightning', 'verl/dataset.py',
                  dict(RLHFDataset=torch.utils.data.Dataset, HuggingFaceDataset=Dataset, torch=torch)).LoadedDataset
    train, dev = (loaded(Dataset.from_parquet(str(p)).to_list()) for p in (train_path, dev_path))
    assert not train.filter_overlong_prompts
    config = OmegaConf.create({'data': {'shuffle': shuffle, 'seed': contract.streams(seed, contract.ALGORITHMS[0])['paired_question_order'],
        'train_batch_size': 8, 'val_batch_size': None, 'validation_shuffle': False, 'dataloader_num_workers': 8},
        'global_profiler': {'steps': None}, 'trainer': {'total_epochs': 100, 'total_training_steps': steps,
        'val_before_train': False, 'test_freq': -1, 'save_freq': 20, 'project_name': 'offline',
        'experiment_name': 'loader', 'logger': []}})
    sampler = load('verl', 'trainer/main_ppo.py', {}, ['create_rl_sampler'])
    collate = load('verl', 'utils/dataset/rl_dataset.py', dict(torch=torch, np=np, defaultdict=defaultdict), ['collate_fn'])
    trainer = NS(config=config, is_async=False, _load_checkpoint=Mock(), _save_checkpoint=Mock(),
                 resource_pool_manager=NS(get_n_gpus=lambda: 0), checkpoint_manager=Mock())
    create = load('verl', 'trainer/ppo/ray_trainer.py', dict(StatefulDataLoader=StatefulDataLoader, OmegaConf=OmegaConf,
                  open_dict=open_dict), ['_create_dataloader'], 'RayPPOTrainer')
    with patch.dict('sys.modules', {'verl.trainer.main_ppo': NS(create_rl_sampler=sampler.create_rl_sampler, create_rl_dataset=None)}):
        create._create_dataloader(trainer, train, dev, collate.collate_fn, sampler.create_rl_sampler(config.data, train))
    assert trainer.train_dataloader.drop_last and trainer.train_dataloader.batch_size == 8
    batches = []
    def observe(*args):
        batches.append(trainer._next_train_batch_dict_for_rollout()['case_id'].tolist())
        assert trainer.epoch == 1
        return {}, None
    trainer._train_step = observe
    methods = load('agentlightning', 'verl/trainer.py', dict(Tracking=Mock(), OmegaConf=OmegaConf, tqdm=Mock(),
        marked_timer=lambda *a, **kw: nullcontext(), compute_timing_metrics=lambda **kw: {}, pprint=lambda x: None),
        ['fit', '_next_train_batch_dict', '_next_train_batch_dict_for_rollout'], 'AgentLightningRayPPOTrainer')
    for name in ('_next_train_batch_dict', '_next_train_batch_dict_for_rollout'):
        setattr(trainer, name, getattr(methods, name).__get__(trainer))
    methods.fit(trainer)
    assert trainer.global_steps == steps and len(batches) == steps
    trainer._save_checkpoint.assert_called_once()
    trainer._train_dataloader_iter._shutdown_workers()
    return batches

import os

import numpy as np
import torch
import torch.nn as nn
import contextlib
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from util.tools import get_timepc, get_net_params
from util.logger import get_logger
from src.model import get_model
from src.data import get_dataloader_test


class BasicTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.master = cfg.master
        self.logger = get_logger(self.cfg) if self.master else None
        self.local_rank, self.rank, self.world_size = cfg.local_rank, cfg.rank, cfg.world_size
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        # prepare network
        self.model = get_model(cfg)
        self.model.to(self.device)
        self.model.eval()

        self.logger.log_msg("- - - > Built model") if self.master else None
        model_params, frozen_params = get_net_params(self.model)
        self.cfg.model.params = model_params
        self.cfg.model.frozen_params = frozen_params

        if cfg.dist and cfg.tester.sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.log_msg("- - - > Convert to SyncBatchNorm") if self.master else None
        if cfg.dist:
            self.model = NativeDDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
            self.logger.log_msg("- - - > NativeDDP") if self.master else None

        self.autocast = autocast() if cfg.tester.amp else contextlib.nullcontext()

        # data
        self.test_loader = get_dataloader_test(cfg)

        # resume
        if self.cfg.tester.resume_ckpt is None:
            default_ckpt_path = f"checkpoints/4cls_ckpt/{cfg.model.name.lower()}.pth"
            self.cfg.tester.resume_ckpt = default_ckpt_path
        assert self.cfg.tester.resume_ckpt is not None
        self.load_weight()

        # log cfg
        self.logger.log_cfg(cfg) if self.master else None

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.device)
        self.target = inputs['target'].to(self.device)

    def forward(self):
        with self.autocast:
            self.output = self.model(self.input)

    def update_cur_log(self):
        pass

    def test(self):
        pass

    def run(self):
        t_s = get_timepc()
        self.test()
        t_e = get_timepc()
        t_cost = t_e - t_s
        self.cur_logs = "Task Done!\tTime Cost:{:.3f}s".format(t_cost)
        self.logger.log_msg(self.cur_logs) if self.master else None
        if self.cfg.dist:
            dist.destroy_process_group()

    def load_weight(self):
        ckpt_dict = torch.load(self.cfg.tester.resume_ckpt, map_location=self.device)

        # model
        model_state_dict = self.model.state_dict()
        model_ckpt_state_dict = ckpt_dict['model']

        model_ckpt_state_dict_precess = {}
        for k, v in model_ckpt_state_dict.items():
            if k.startswith('module.') and not self.cfg.dist:
                k = k[7:]
                model_ckpt_state_dict_precess[k] = v
            elif self.cfg.dist and not k.startswith('module.'):
                k = 'module.' + k
                model_ckpt_state_dict_precess[k] = v
            else:
                model_ckpt_state_dict_precess[k] = v

        fail_load_keys, temp_dict = [], {}
        for k, v in model_ckpt_state_dict_precess.items():
            if k in model_state_dict.keys() and np.shape(model_state_dict[k]) == np.shape(v):
                temp_dict[k] = v
            elif k in model_state_dict.keys():
                fail_load_keys.append(k)

        model_state_dict.update(temp_dict)
        self.model.load_state_dict(model_state_dict)
        self.logger.log_msg("RESUME\n[MODEL]\t[Success to loaded: {}\t{}]\n[MODEL]\t[Fail to loaded: {}\t{}]\n".format(len(temp_dict), list(temp_dict.keys()), len(fail_load_keys), fail_load_keys)) if self.master else None

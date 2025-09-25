import os
import numpy as np
import torch
import torch.nn as nn
import contextlib
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from util.tools import distribute_bn, get_timepc, get_net_params
from util.logger import get_logger
from src.model import get_model
from src.data import get_dataloader
from src.optim import get_optimizer, get_scheduler
from src.loss import get_loss_fn


class BasicTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.master = cfg.master
        self.logger = get_logger(self.cfg) if self.master else None
        self.writer = SummaryWriter(log_dir=cfg.logger.path, comment='') if self.master else None
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

        if cfg.dist and cfg.trainer.sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.log_msg("- - - > Convert to SyncBatchNorm") if self.master else None
        if cfg.dist:
            self.model = NativeDDP(self.model, device_ids=[self.local_rank], find_unused_parameters=False)
            self.logger.log_msg("- - - > NativeDDP") if self.master else None

        self.autocast = autocast() if cfg.trainer.amp else contextlib.nullcontext()
        self.scaler = GradScaler() if cfg.trainer.amp else None

        # data
        self.train_loader, self.test_loader = get_dataloader(cfg)

        # optimizer, scheduler
        self.optim = get_optimizer(self.model, cfg)
        self.warmup_scheduler, self.scheduler = get_scheduler(self.optim, cfg)

        # loss
        self.loss_fn = get_loss_fn(cfg)
        self.loss_fn.to(self.device)

        # basic configs
        self.total_epoch = cfg.total_epochs
        self.total_iter = cfg.trainer.total_iter
        self.epoch_now = 0
        self.iter_now = 0
        self.iter_warmup = cfg.trainer.warmup_iter
        self.warmup = True if self.warmup_scheduler is not None else False
        self.is_best = False

        # log cfg
        self.logger.log_cfg(cfg) if self.master else None

        # resume
        if self.cfg.trainer.resume_ckpt is not None:
            self.load_weight()

    def reset(self, is_train: bool = False):
        if is_train:
            self.model.train()
            self.logger.log_msg(
                "[Epoch:{}/{}]\nStart Training...".format(self.epoch_now, self.total_epoch)) if self.master else None
        else:
            self.model.eval()
            self.logger.log_msg("Start Testing...".format(self.epoch_now, self.total_epoch)) if self.master else None

    def check_bn(self):
        if hasattr(self.model, 'module'):
            self.model.module.check_bn() if hasattr(self.model.module, 'check_bn') else None
        else:
            self.model.check_bn() if hasattr(self.model, 'check_bn') else None

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.device)
        self.target = inputs['target'].to(self.device)

    def scheduler_step(self):
        if self.scheduler is None:
            pass
        elif self.iter_now < self.iter_warmup and self.warmup:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

    def forward(self):
        with self.autocast:
            self.output = self.model(self.input)

    def backward(self, loss, optim):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            loss.backward()
            optim.step()

    def get_loss(self):
        return self.loss_fn(self.output, self.target)

    def optimize_parameters(self):
        self.forward()

        self.loss = self.get_loss()
        if not isinstance(self.loss, dict):
            self.loss = dict(loss=self.loss)
        self.backward(self.loss['loss'], self.optim)

        self.writer.add_scalar('loss', self.loss['loss'], self.iter_now) if self.master else None
        self.writer.add_scalar('lr', self.optim.state_dict()['param_groups'][0]['lr'],
                               self.iter_now) if self.master else None

    def update_cur_log(self):
        self.lr = self.optim.state_dict()['param_groups'][0]['lr']
        self.cur_log = "[Train: {:.2f}% | {}/{}]\t[Loss: {:.4f}]\t[LR: {:.6f}]".format(
            100 * self.iter_now / self.total_iter, self.iter_now, self.total_iter, self.loss['loss'], self.lr)

    def train_one_epoch(self):
        t_s = get_timepc()
        loss_per_epoch = 0.0
        self.train_loader.sampler.set_epoch(int(self.epoch_now)) if self.cfg.dist else None
        self.reset(is_train=True)
        self.check_bn()

        for batch_data in self.train_loader:
            self.iter_now += 1
            # get data
            self.set_input(batch_data)

            # update params
            self.optim.zero_grad()
            self.optimize_parameters()

            if self.iter_now % self.cfg.logger.log_freq == 0:
                # update logs
                self.update_cur_log()
                self.logger.log_msg(self.cur_log) if self.master else None

            # update lr
            self.scheduler_step()
            loss_per_epoch += self.loss["loss"].item()

        if self.cfg.dist and not self.cfg.trainer.sync_bn:
            distribute_bn(self.model, self.world_size, self.cfg.trainer.dist_bn)

        t_e = get_timepc()
        t_cost = t_e - t_s
        avg_loss_per_epoch = loss_per_epoch / len(self.train_loader)
        self.cur_logs = "Train Done!\tTime Cost:{:.3f}s\tTotal Loss:{:.3f}\tAvg Loss:{:.3f}".format(t_cost, loss_per_epoch, avg_loss_per_epoch)
        self.logger.log_msg(self.cur_logs) if self.master else None

    def test_model_single(self):
        pass

    def test_model_multi(self):
        pass

    def test_model_single_dist(self):
        pass

    def test_model_multi_dist(self):
        pass

    def test_model(self):
        if not self.cfg.data.is_multi_loader:
            self.test_model_single()
        else:
            self.test_model_multi()

    def train(self):
        for i in range(self.total_epoch):
            self.epoch_now += 1

            if self.epoch_now > self.total_epoch or self.iter_now > self.total_iter:
                self.save_ckpt()
                self.writer.close() if self.master else None
                break

            self.train_one_epoch()

            # test and save ckpt
            if self.epoch_now >= self.cfg.test_start_epoch:
                self.test_model()
            self.save_ckpt()
        self.writer.close() if self.master else None

    def run(self):
        t_s = get_timepc()
        self.train()
        t_e = get_timepc()
        t_cost = t_e - t_s
        self.cur_logs = "Task Done!\tTime Cost:{:.3f}s".format(t_cost)
        self.logger.log_msg(self.cur_logs) if self.master else None
        if self.cfg.dist:
            dist.destroy_process_group()

    def save_ckpt(self):
        if self.master:
            ckpt_infos = {
                "model": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "iter": self.iter_now,
                "epoch": self.epoch_now,
            }
            dir_name = os.path.join(self.cfg.trainer.ckpt_dir, str(self.cfg.model.name), self.cfg.task_start_time)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            if self.epoch_now % self.cfg.trainer.save_freq == 0:
                base_name = "latest_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.logger.log_msg("checkpoint saved to {}".format(save_path)) if self.master else None
            if self.is_best:
                base_name = "best_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.is_best = False
                self.logger.log_msg("checkpoint saved to {}".format(save_path)) if self.master else None

    def load_weight(self):
        ckpt_dict = torch.load(self.cfg.trainer.resume_ckpt, map_location=f'cuda:{self.local_rank}')

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
        self.logger.log_msg(
            "RESUME\n[MODEL]\t[Success to loaded: {}\t{}]\n[MODEL]\t[Fail to loaded: {}\t{}]\n".format(len(temp_dict), list(temp_dict.keys()), len(fail_load_keys), fail_load_keys)) if self.master else None

        # optimizer
        optimizer_state_dict = ckpt_dict['optimizer']
        self.optim.load_state_dict(optimizer_state_dict)
        self.logger.log_msg("RESUME\n[OPTIMIZER] [Resume from ckpt]\n") if self.master else None

        # other
        self.iter_now = ckpt_dict['iter']
        self.epoch_now = ckpt_dict['epoch']

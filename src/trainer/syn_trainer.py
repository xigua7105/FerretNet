import os
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from util.tools import get_timepc
from sklearn.metrics import accuracy_score, average_precision_score
from .basic_trainer import BasicTrainer


class SynTrainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_acc = float("-inf")
        self.best_ap = float('-inf')
        self.is_best_acc = False
        self.is_best_ap = False

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.device, non_blocking=True)
        self.target = inputs['target'].to(self.device, non_blocking=True).float()

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.target)

    def _gather_list_across_processes(self, local_list):
        if self.cfg.dist:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, list(local_list))
            if self.master:
                merged = []
                for part in gathered:
                    if part:
                        merged.extend(part)
                return merged
            else:
                return []
        else:
            return list(local_list)

    @torch.no_grad()
    def test_model_multi(self):
        t_s = get_timepc()
        self.reset(is_train=False)
        # self.check_bn()

        acc_list, ap_list, r_acc_list, f_acc_list = [], [], [], []

        for k, v in self.test_loader.items():
            local_y_true, local_y_pred = [], []
            _t_s = get_timepc()

            for batch_data in tqdm(v, disable=(not self.master), desc=f"Testing [{k}]", leave=False):
                self.set_input(batch_data)
                self.forward()

                batch_true = self.target.flatten().tolist()
                batch_pred = self.output.sigmoid().flatten().tolist()

                local_y_true.extend(batch_true)
                local_y_pred.extend(batch_pred)

            y_true = self._gather_list_across_processes(local_y_true)
            y_pred = self._gather_list_across_processes(local_y_pred)

            _t_e = get_timepc()
            t_cost = _t_e - _t_s

            if self.master:
                y_true = np.asarray(y_true, dtype=np.float32)
                y_pred = np.asarray(y_pred, dtype=np.float32)

                acc = 100.0 * accuracy_score(y_true, y_pred > 0.5)
                ap = 100.0 * average_precision_score(y_true, y_pred)

                r_acc = 100.0 * accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
                f_acc = 100.0 * accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

                acc_list.append(acc)
                ap_list.append(ap)
                r_acc_list.append(r_acc)
                f_acc_list.append(f_acc)

                self.cur_logs = "[{:^15}]\t[Time Cost:{:.3f}s]\t[Acc:{:.3f}]\t[AP:{:.3f}]\t[R_ACC:{:.3f}]\t[F_ACC:{:.3f}]".format(
                    k, t_cost, acc, ap, r_acc, f_acc
                )
                self.logger.log_msg(self.cur_logs)

        t_e = get_timepc()
        t_cost = t_e - t_s

        if self.master:
            cur_acc = np.mean(acc_list)
            cur_ap = np.mean(ap_list)
            self.cur_logs = "Test Done!\tTime Cost:{:.3f}s\tAcc:{:.3f}\tAP:{:.3f}\tR_ACC:{:.3f}\tF_ACC:{:.3f}\t".format(
                t_cost, cur_acc, cur_ap, np.mean(r_acc_list), np.mean(f_acc_list)
            )
            self.logger.log_msg(self.cur_logs)

            if self.best_acc < cur_acc:
                self.best_acc = cur_acc
                self.is_best_acc = True
            if self.best_ap < cur_ap:
                self.best_ap = cur_ap
                self.is_best_ap = True

    def save_ckpt(self):
        if self.master:
            ckpt_infos = {
                "model": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "iter": self.iter_now,
                "epoch": self.epoch_now,
            }
            print()
            dir_name = os.path.join(self.cfg.trainer.ckpt_dir, str(self.cfg.model.name), self.cfg.task_start_time)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            if self.epoch_now % self.cfg.trainer.save_freq == 0:
                base_name = "latest_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.logger.log_msg("checkpoint saved to {}".format(save_path))
            if self.is_best_acc:
                base_name = "best_acc_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.is_best_acc = False
                self.logger.log_msg("checkpoint saved to {}".format(save_path))
            if self.is_best_ap:
                base_name = "best_ap_ckpt.pth"
                save_path = os.path.join(str(dir_name), base_name)
                torch.save(ckpt_infos, save_path)
                self.is_best_ap = False
                self.logger.log_msg("checkpoint saved to {}".format(save_path))

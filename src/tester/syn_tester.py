import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from util.tools import get_timepc
from .basic_tester import BasicTester
from sklearn.metrics import accuracy_score, average_precision_score


class SynTester(BasicTester):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_input(self, inputs):
        self.input = inputs['img'].to(self.device, non_blocking=True)
        self.target = inputs['target'].to(self.device, non_blocking=True).float()

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
    def test(self):
        t_s = get_timepc()

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
            self.cur_logs = "Test Done!\tTime Cost:{:.3f}s\tAcc:{:.3f}\tAP:{:.3f}\tR_ACC:{:.3f}\tF_ACC:{:.3f}\t".format(
                t_cost, np.mean(acc_list), np.mean(ap_list), np.mean(r_acc_list), np.mean(f_acc_list)
            )
            self.logger.log_msg(self.cur_logs)

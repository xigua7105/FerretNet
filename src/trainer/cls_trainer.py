import torch
from util.metric import topk_accuracy
from util.tools import get_timepc
from .basic_trainer import BasicTrainer


class CLSTrainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_top1 = float("-inf")

    @torch.no_grad()
    def test_model_single(self):
        t_s = get_timepc()
        self.reset(is_train=False)
        self.check_bn()
        top_1, top_5 = 0.0, 0.0
        for batch_data in self.test_loader:
            self.set_input(batch_data)
            self.forward()

            _top_1, _top_5 = topk_accuracy(self.output, self.target)
            top_1 += _top_1
            top_5 += _top_5

        top_1 = 100 * top_1 / self.cfg.data.test_length
        top_5 = 100 * top_5 / self.cfg.data.test_length
        t_e = get_timepc()

        t_cost = t_e - t_s
        self.cur_logs = "Test Done!\tTime Cost:{:.3f}s\tTop_1:{:.3f}%\tTop_5:{:.3f}%".format(t_cost, top_1, top_5)
        self.logger.log_msg(self.cur_logs) if self.master else None
        if self.best_top1 < top_1:
            self.best_top1 = top_1
            self.is_best = True

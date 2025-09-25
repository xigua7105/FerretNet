import os
import sys
import logging


class Logger:
    def __init__(self, cfg):
        logdir = os.path.join(str(cfg.logger.dir), str(cfg.model.name), cfg.task_start_time)
        os.makedirs(logdir, exist_ok=True)
        cfg.logger.path = logdir

        log_file_name = os.path.join(logdir, 'logs.txt')

        self.logger = logging.getLogger('mylogger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        fh = logging.FileHandler(log_file_name, mode='a+')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(ch)

    def log_msg(self, message):
        self.logger.info(message)

    def log_cfg(self, cfg):
        cfg_dict = cfg.get_cfg_dict()
        if hasattr(cfg_dict, '__dict__'):
            cfg_dict = cfg_dict.__dict__
        key_max_length = max(list(map(len, cfg_dict.keys())))

        cfg_str = ''
        temp_k = ''
        for k, v in cfg_dict.items():
            if k.split('|')[0] != temp_k:
                temp_k = k.split('|')[0]
                cfg_str += '-------------------------------------------------\n'
            cfg_str += ('{'+':<{}'.format(key_max_length)+'} : {'+':<{}'.format(key_max_length)+'}').format(k, str(v))
            cfg_str += '\n'
        cfg_str = cfg_str.strip()

        self.logger.info(f'- - - > Logging configs\n{cfg_str}\n-------------------------------------------------\n')


def get_logger(cfg):
    return Logger(cfg)

import argparse
from util import get_cfg, init_cfg
from datetime import datetime
from src.tester import get_tester
from src.trainer import get_trainer


def get_runner(cfg):
    assert cfg.mode in ['train', 'test']
    if cfg.mode == 'train':
        return get_trainer(cfg)
    return get_tester(cfg)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str)
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    cfg = get_cfg(**parser.parse_args().__dict__)
    cfg.__setattr__("task_start_time", datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))
    init_cfg(cfg)
    runner = get_runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()

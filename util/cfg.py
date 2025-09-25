import yaml
from argparse import Namespace


class Config:
    def __init__(self, mode='test', **kwargs):
        super().__init__()
        self.mode = mode

    def get_cfg_dict(self):
        cfg_dict = dict()
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                for k, v in value.__dict__.items():
                    cfg_dict[f'{key} | {k}'] = v
            else:
                cfg_dict[key] = value
        return cfg_dict


def read_yaml_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Not Found [{file_path}].")
    except yaml.YAMLError as e:
        print(f"Error: {e}")
    return None


def get_cfg(cfg_path=None, **kwargs):

    base_cfg = Config(**kwargs)
    if cfg_path is None:
        return base_cfg

    update_cfg = read_yaml_config(cfg_path)
    for k, v in update_cfg.items():
        if isinstance(v, dict):
            base_cfg.__setattr__(k, Namespace())
            for _k, _v in v.items():
                base_cfg.__dict__[k].__setattr__(_k, _v)
        else:
            base_cfg.__setattr__(k, v)

    return base_cfg
